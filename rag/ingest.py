"""PDF -> pages -> chunks. Chunks never cross page boundaries, so every citation is an exact page."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")


def parse_doc_meta(path: Path) -> dict[str, Any]:
    """Derive year / type from the file name. Wrong guesses only affect soft year boosting, never citations."""
    name = path.stem
    years = _YEAR.findall(name)
    low = name.lower()
    if "proxy" in low:
        doc_type = "proxy statement"
    elif "annual" in low or "10-k" in low or "10k" in low:
        doc_type = "annual report"
    else:
        doc_type = "document"
    return {"doc": path.name, "year": int(years[-1]) if years else None, "doc_type": doc_type}


def _table_to_markdown(rows: list[list[Any]]) -> str:
    clean = [[("" if c is None else re.sub(r"\s+", " ", str(c)).strip()) for c in r] for r in rows]
    clean = [r for r in clean if any(r)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def looks_tabular(text: str, min_numbers: int = 12) -> bool:
    """Identify potentially tabular pages, including financial tables."""
    return len(_NUM.findall(text)) >= min_numbers

def extract_pdf(path_str: str, tables: bool = True, ocr: bool = False, sort: bool = False) -> dict[str, Any]:
    """Top-level (picklable) worker. Returns pages with text and markdown tables."""
    import pymupdf

    pages: list[dict[str, Any]] = []
    empty = 0
    with pymupdf.open(path_str) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=sort) or ""
            used_ocr = False
            if len(text.strip()) < 20:
                if ocr:
                    try:
                        text = page.get_textpage_ocr(full=True).extractText()
                        used_ocr = True
                    except Exception as exc:  # Tesseract missing etc.
                        log.warning("OCR unavailable for %s p%d: %s", path_str, i, exc)
                if len(text.strip()) < 20:
                    empty += 1
            tbls: list[str] = []
            if tables and looks_tabular(text):
                try:
                    for t in page.find_tables().tables:
                        md = _table_to_markdown(t.extract())
                        if md.count("\n") >= 2:
                            tbls.append(md)
                except Exception as exc:
                    log.debug("table extraction failed %s p%d: %s", path_str, i, exc)
            pages.append({"page": i, "text": text, "tables": tbls, "ocr": used_ocr})
    return {"pages": pages, "empty_pages": empty}


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Line-aware word-window chunking with overlap; keeps line breaks (helps tables/lists)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for ln in lines:
        w = len(ln.split())
        if cur and cur_words + w > size:
            chunks.append("\n".join(cur))
            carry: list[str] = []
            cw = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if cw + pw > overlap:
                    break
                carry.insert(0, prev)
                cw += pw
            cur, cur_words = carry, cw
        cur.append(ln)
        cur_words += w
    if cur and (len(chunks) == 0 or "\n".join(cur) != chunks[-1]):
        tail = "\n".join(cur)
        # avoid a tiny tail that is only overlap
        if not chunks or cur_words > overlap:
            chunks.append(tail)
    return chunks


def split_table(md: str, size: int) -> list[str]:
    lines = md.splitlines()
    header, body = lines[:2], lines[2:]
    out, cur, words = [], [], 0
    for ln in body:
        w = len(ln.split())
        if cur and words + w > size:
            out.append("\n".join(header + cur))
            cur, words = [], 0
        cur.append(ln)
        words += w
    if cur:
        out.append("\n".join(header + cur))
    return out


def build_chunks(meta: dict[str, Any], pages: list[dict[str, Any]], size: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    label = f"{meta['doc_type']} {meta['year'] or ''}".strip()
    for pg in pages:
        pieces = [("text", t) for t in split_text(pg["text"], size, overlap)]
        for md in pg["tables"]:
            pieces += [("table", t) for t in split_table(md, size)]
        for kind, t in pieces:
            chunks.append(
                {
                    "doc": meta["doc"],
                    "year": meta["year"],
                    "doc_type": meta["doc_type"],
                    "page": pg["page"],
                    "kind": kind,
                    "text": t,
                    "embed_text": f"{label} | {meta['doc']} | page {pg['page']}\n{t}",
                }
            )
    return chunks
