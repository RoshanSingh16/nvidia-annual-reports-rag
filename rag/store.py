"""Persistent, incremental index store.

Per-PDF cache (keyed by file SHA-256 + chunk/embedding signature) holds chunks and embeddings, so a restart
or a newly added PDF never re-embeds untouched documents. The combined FAISS index is also persisted and
only rebuilt when the set of documents changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np

from .config import Settings
from .bm25 import SparseBM25
from .embeddings import make_embedder
from .ingest import build_chunks, extract_pdf, parse_doc_meta
import re

log = logging.getLogger(__name__)
_TOK = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
ProgressCb = Callable[[float, str], None]


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class SyncReport:
    processed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    seconds_extract: float = 0.0
    seconds_embed: float = 0.0
    seconds_total: float = 0.0


class IndexStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.root = settings.index_dir
        (self.root / "docs").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict = self._load_manifest()
        self._embedder = None
        # runtime structures
        self.chunks: list[dict] = []
        self.emb: np.ndarray = np.zeros((0, 0), np.float32)
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25: Optional[SparseBM25] = None
        self.last_report: Optional[SyncReport] = None

    # ---------- helpers ----------
    @property
    def embedder(self):
        if self._embedder is None:  # one model instance per process
            self._embedder = make_embedder(self.s.embed_backend, self.s.embed_model)
        return self._embedder

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                log.warning("Corrupt manifest; rebuilding from scratch")
        return {"docs": {}}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1))
        tmp.replace(self.manifest_path)

    def _doc_files(self, key: str) -> tuple[Path, Path]:
        return self.root / "docs" / f"{key}.json", self.root / "docs" / f"{key}.npy"

    def list_pdfs(self) -> list[Path]:
        return sorted(p for p in self.s.pdf_dir.glob("*.pdf"))

    # ---------- sync ----------
    def sync(self, progress: Optional[ProgressCb] = None) -> SyncReport:
        t0 = time.perf_counter()
        rep = SyncReport()
        sig = self.s.signature()
        pdfs = self.list_pdfs()
        todo: list[tuple[Path, str]] = []
        active: dict[str, dict] = {}
        for i, p in enumerate(pdfs):
            st = p.stat()
            prev = self.manifest["docs"].get(p.name, {})
            if prev.get("size") == st.st_size and prev.get("mtime") == st.st_mtime:
                sha = prev["sha256"]  # unchanged stat -> skip rehash
            else:
                sha = _sha256(p)
            key = f"{sha[:16]}-{sig}"
            js, npy = self._doc_files(key)
            if js.exists() and npy.exists():
                rep.reused.append(p.name)
                active[p.name] = {**prev, "sha256": sha, "size": st.st_size, "mtime": st.st_mtime, "key": key}
            else:
                todo.append((p, sha))
            if progress:
                progress(0.05 * (i + 1) / max(len(pdfs), 1), f"Checking {p.name}")
        rep.removed = [n for n in self.manifest["docs"] if n not in {p.name for p in pdfs}]

        if todo:
            self._process(todo, active, rep, sig, progress)
        self.manifest = {"docs": active, "signature": sig, "updated": time.time()}
        self._save_manifest()
        self._build_runtime(progress)
        rep.seconds_total = time.perf_counter() - t0
        self.last_report = rep
        self.manifest["last_sync"] = {
            "processed": len(rep.processed), "reused": len(rep.reused), "failed": rep.failed,
            "seconds_extract": rep.seconds_extract, "seconds_embed": rep.seconds_embed, "seconds_total": rep.seconds_total,
        }
        self._save_manifest()
        return rep

    def _process(self, todo, active, rep: SyncReport, sig: str, progress) -> None:
        # 1) parallel extraction (CPU-bound, per process)
        t0 = time.perf_counter()
        extracted: dict[str, dict] = {}
        n = len(todo)
        def _record(k, p, getter):
            try:
                extracted[p.name] = getter()
            except Exception as exc:  # corrupt/encrypted PDF must not kill the whole sync
                log.error("Extraction failed for %s: %s", p.name, exc)
                rep.failed[p.name] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress(0.05 + 0.35 * k / n, f"Extracted {p.name}")

        if self.s.workers <= 1 or n == 1:  # serial path (also the safe fallback on Windows/Streamlit)
            for k, (p, sha) in enumerate(todo, start=1):
                _record(k, p, lambda p=p: extract_pdf(str(p), self.s.extract_tables, self.s.ocr, self.s.sort_text))
        else:
            with ProcessPoolExecutor(max_workers=min(self.s.workers, n)) as ex:
                futs = {ex.submit(extract_pdf, str(p), self.s.extract_tables, self.s.ocr, self.s.sort_text): (p, sha) for p, sha in todo}
                for k, fut in enumerate(as_completed(futs), start=1):
                    p, _ = futs[fut]
                    _record(k, p, fut.result)
        rep.seconds_extract = time.perf_counter() - t0
        # 2) chunk + embed in large batches (single model instance)
        t1 = time.perf_counter()
        for k, (p, sha) in enumerate(todo, start=1):
            if p.name not in extracted:
                continue
            ex_res = extracted[p.name]
            meta = parse_doc_meta(p)
            chunks = build_chunks(meta, ex_res["pages"], self.s.chunk_words, self.s.chunk_overlap)
            if not chunks:
                rep.failed[p.name] = "No extractable text (scanned PDF? set OCR=1 and install Tesseract)"
                continue
            vecs = self.embedder.embed_documents([c["embed_text"] for c in chunks], batch_size=64)
            key = f"{sha[:16]}-{sig}"
            js, npy = self._doc_files(key)
            for c in chunks:
                c.pop("embed_text", None)
            np.save(npy, vecs)
            js.write_text(json.dumps({"meta": meta, "chunks": chunks}))
            st = p.stat()
            active[p.name] = {
                "sha256": sha, "size": st.st_size, "mtime": st.st_mtime, "key": key, **meta,
                "pages": len(ex_res["pages"]), "empty_pages": ex_res["empty_pages"], "chunks": len(chunks),
                "table_chunks": sum(c["kind"] == "table" for c in chunks),
            }
            rep.processed.append(p.name)
            if progress:
                progress(0.4 + 0.5 * k / len(todo), f"Embedded {p.name} ({len(chunks)} chunks)")
        rep.seconds_embed = time.perf_counter() - t1

    # ---------- runtime ----------
    def _fingerprint(self) -> str:
        keys = sorted(d["key"] for d in self.manifest["docs"].values())
        return hashlib.sha1((self.s.index_type + "|" + ",".join(keys)).encode()).hexdigest()[:16]

    def _build_runtime(self, progress: Optional[ProgressCb] = None) -> None:
        chunks: list[dict] = []
        mats: list[np.ndarray] = []
        for name, d in self.manifest["docs"].items():
            js, npy = self._doc_files(d["key"])
            data = json.loads(js.read_text())
            chunks.extend(data["chunks"])
            mats.append(np.load(npy))
        self.chunks = chunks
        if not chunks:
            self.emb, self.faiss_index, self.bm25 = np.zeros((0, 0), np.float32), None, None
            return
        self.emb = np.vstack(mats).astype(np.float32)
        fp = self._fingerprint()
        fpath = self.root / f"combined-{fp}.faiss"
        index = None
        if fpath.exists():
            try:
                index = faiss.read_index(str(fpath))
                if index.ntotal != len(chunks):
                    index = None
            except Exception:
                index = None
        if index is None:
            d = self.emb.shape[1]
            if self.s.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 100
                index.hnsw.efSearch = 64
            else:
                index = faiss.IndexFlatIP(d)
            index.add(self.emb)
            for old in self.root.glob("combined-*.faiss"):
                old.unlink(missing_ok=True)
            faiss.write_index(index, str(fpath))
        self.faiss_index = index
        stem = self.root / f"bm25-{fp}"
        bm = None
        if Path(str(stem) + ".npz").exists():
            try:
                bm = SparseBM25.load(stem)
                if bm.matrix.shape[0] != len(chunks):
                    bm = None
            except Exception:
                bm = None
        if bm is None:
            bm = SparseBM25.build([tokenize(c["text"] + " " + c["doc"]) for c in chunks])
            for old in self.root.glob("bm25-*"):
                old.unlink(missing_ok=True)
            bm.save(stem)
        self.bm25 = bm
        if progress:
            progress(1.0, f"Index ready: {len(chunks):,} chunks")

    def load(self) -> bool:
        """Load existing index from disk without touching PDFs. Returns True if something was loaded."""
        if not self.manifest.get("docs") or self.manifest.get("signature") != self.s.signature():
            return False
        try:
            self._build_runtime()
        except Exception as exc:
            log.warning("Could not load persisted index: %s", exc)
            return False
        return bool(self.chunks)

    # ---------- stats ----------
    def stats(self) -> dict:
        docs = self.manifest.get("docs", {})
        return {
            "pdfs_on_disk": len(self.list_pdfs()),
            "indexed_docs": len(docs),
            "pages": sum(d.get("pages", 0) for d in docs.values()),
            "empty_pages": sum(d.get("empty_pages", 0) for d in docs.values()),
            "chunks": len(self.chunks),
            "table_chunks": sum(d.get("table_chunks", 0) for d in docs.values()),
            "index_mb": round(self.emb.nbytes / 1e6, 1) if self.emb.size else 0.0,
            "last_sync": self.manifest.get("last_sync"),
        }
