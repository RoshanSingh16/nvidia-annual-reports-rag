"""Hybrid retrieval (dense FAISS + BM25, fused with RRF), optional cross-encoder rerank, caching, timing."""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Settings
from .store import IndexStore, tokenize

log = logging.getLogger(__name__)
_YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")
RRF_K = 60


@dataclass
class Hit:
    idx: int
    doc: str
    page: int
    year: Optional[int]
    doc_type: str
    kind: str
    text: str
    dense: float          # cosine similarity to the query (always computed)
    bm25: float           # raw BM25 score (0 if not run)
    score: float          # final ranking score (RRF or rerank)
    rank: int = 0


@dataclass
class Retrieval:
    query: str
    mode: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    best_dense: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


class LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


class Retriever:
    def __init__(self, store: IndexStore, settings: Settings):
        self.store, self.s = store, settings
        self._qcache = LRU(512)      # query embeddings
        self._rcache = LRU(256)      # full retrieval results
        self._reranker = None
        self._rerank_failed = False

    # --- helpers ---
    def _embed_query(self, q: str) -> np.ndarray:
        key = hashlib.md5(q.encode()).hexdigest()
        if key in self._qcache:
            return self._qcache[key]
        v = self.store.embedder.embed_query(q)
        self._qcache.put(key, v)
        return v

    def _get_reranker(self):
        if self._reranker is None and not self._rerank_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._reranker = TextCrossEncoder(model_name=self.s.rerank_model)
            except Exception as exc:
                log.warning("Reranker unavailable (%s); continuing without rerank", exc)
                self._rerank_failed = True
        return self._reranker

    @staticmethod
    def _norm_text_key(t: str) -> str:
        return hashlib.md5(re.sub(r"\W+", "", t.lower())[:300].encode()).hexdigest()

    # --- main ---
    def search(self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None,
               rerank: Optional[bool] = None, use_cache: bool = True) -> Retrieval:
        st = self.store
        top_k = top_k or self.s.top_k
        mode = mode or self.s.mode
        rerank = self.s.rerank if rerank is None else rerank
        if not st.chunks:
            return Retrieval(query, mode, [], {"total": 0.0})
        ckey = (query.strip().lower(), top_k, mode, bool(rerank), len(st.chunks))
        if use_cache and ckey in self._rcache:
            cached: Retrieval = self._rcache[ckey]
            return Retrieval(cached.query, cached.mode, cached.hits, {"total": 0.05, "cache": 0.05}, True, cached.best_dense)

        T: dict[str, float] = {}
        t_all = time.perf_counter()
        n = len(st.chunks)
        cand = min(self.s.candidate_k, n)

        t = time.perf_counter()
        qv = self._embed_query(query)  # also used to show a cosine score for every hit, whatever the mode
        T["embed"] = (time.perf_counter() - t) * 1e3

        dense_ids: list[int] = []
        if mode in {"hybrid", "dense"}:
            t = time.perf_counter()
            _, ids = st.faiss_index.search(qv[None, :].astype(np.float32), cand)
            dense_ids = [int(i) for i in ids[0] if i >= 0]
            T["dense"] = (time.perf_counter() - t) * 1e3

        bm_scores: Optional[np.ndarray] = None
        bm_ids: list[int] = []
        if mode in {"hybrid", "bm25"}:
            t = time.perf_counter()
            bm_scores = st.bm25.get_scores(tokenize(query))
            part = np.argpartition(-bm_scores, cand - 1)[:cand] if cand < n else np.arange(n)
            bm_ids = [int(i) for i in part[np.argsort(-bm_scores[part])] if bm_scores[i] > 0]
            T["bm25"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        fused: dict[int, float] = {}
        for lst in (dense_ids, bm_ids):
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        years = {int(y) for y in _YEAR.findall(query)}
        if years:  # soft boost (fiscal-year naming may differ from file-name year, so never a hard filter)
            for i in list(fused):
                c = st.chunks[i]
                if c["year"] in years or any(str(y) in c["text"] for y in years):
                    fused[i] *= 1.25
        ranked = sorted(fused, key=fused.get, reverse=True)
        T["fuse"] = (time.perf_counter() - t) * 1e3

        # optional rerank over the fused candidate pool
        scores = dict(fused)
        if rerank and ranked:
            rr = self._get_reranker()
            if rr is not None:
                t = time.perf_counter()
                pool = ranked[: min(30, len(ranked))]
                rs = list(rr.rerank(query, [st.chunks[i]["text"] for i in pool]))
                scores = {i: float(s) for i, s in zip(pool, rs)}
                ranked = sorted(pool, key=scores.get, reverse=True)
                T["rerank"] = (time.perf_counter() - t) * 1e3

        # dedupe near-identical chunks, then guarantee coverage when several years are named
        seen, uniq = set(), []
        for i in ranked:
            k = self._norm_text_key(st.chunks[i]["text"])
            if k not in seen:
                seen.add(k)
                uniq.append(i)
        chosen: list[int] = []
        if len(years) > 1:
            quota = max(1, top_k // (2 * len(years)))
            for y in sorted(years):
                chosen += [i for i in uniq if st.chunks[i]["year"] == y and i not in chosen][:quota]
        chosen += [i for i in uniq if i not in chosen]
        chosen = chosen[:top_k]
        if len(years) > 1:
            chosen.sort(key=lambda i: -scores.get(i, 0.0))

        sims = st.emb[chosen] @ qv if chosen else np.array([])
        hits = []
        for r, (i, sim) in enumerate(zip(chosen, sims), start=1):
            c = st.chunks[i]
            hits.append(Hit(i, c["doc"], c["page"], c["year"], c["doc_type"], c["kind"], c["text"], float(sim),
                            float(bm_scores[i]) if bm_scores is not None else 0.0, float(scores.get(i, 0.0)), r))
        T["total"] = (time.perf_counter() - t_all) * 1e3
        res = Retrieval(query, mode, hits, T, False, float(max((h.dense for h in hits), default=0.0)))
        if use_cache:
            self._rcache.put(ckey, res)
        return res
