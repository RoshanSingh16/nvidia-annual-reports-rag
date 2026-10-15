"""Central configuration. Everything is overridable via environment variables / .env."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


def _path(name: str, default: str) -> Path:
    p = Path(_get(name, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path
    index_dir: Path
    log_dir: Path
    report_dir: Path
    embed_backend: str
    embed_model: str
    chunk_words: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    mode: str
    rerank: bool
    rerank_model: str
    index_type: str
    min_dense_score: float
    max_context_chars: int
    extract_tables: bool
    ocr: bool
    sort_text: bool
    workers: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_s: float
    llm_max_retries: int
    llm_reasoning_effort: str
    price_in_per_m: float
    price_out_per_m: float

    def signature(self) -> str:
        """Hash of everything that changes chunk text or embeddings; keys the on-disk cache."""
        parts = [self.embed_backend, self.embed_model, self.chunk_words, self.chunk_overlap, self.extract_tables, self.ocr, self.sort_text, "v2"]
        return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


def load_settings(**overrides) -> Settings:
    s = Settings(
        pdf_dir=_path("PDF_DIR", "data/pdfs"),
        index_dir=_path("INDEX_DIR", "data/index"),
        log_dir=_path("LOG_DIR", "data/logs"),
        report_dir=_path("REPORT_DIR", "data/reports"),
        embed_backend=_get("EMBED_BACKEND", "fastembed"),
        embed_model=_get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_words=_get("CHUNK_WORDS", 1200, int),
        chunk_overlap=_get("CHUNK_OVERLAP", 200, int),
        top_k=_get("TOP_K", 6, int),
        candidate_k=_get("CANDIDATE_K", 40, int),
        mode=_get("RETRIEVAL_MODE", "hybrid"),
        rerank=_get("RERANK", False, bool),
        rerank_model=_get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        index_type=_get("INDEX_TYPE", "flat"),
        min_dense_score=_get("MIN_DENSE_SCORE", 0.45, float),
        max_context_chars=_get("MAX_CONTEXT_CHARS", 14000, int),
        extract_tables=_get("EXTRACT_TABLES", True, bool),
        ocr=_get("OCR", False, bool),
        sort_text=_get("SORT_TEXT", False, bool),
        workers=_get("WORKERS", 4, int),
        llm_base_url=_get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        llm_model=_get("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
        llm_api_key=_get("LLM_API_KEY", _get("NVIDIA_API_KEY", "")),
        llm_timeout_s=_get("LLM_TIMEOUT_S", 60.0, float),
        llm_max_retries=_get("LLM_MAX_RETRIES", 3, int),
        llm_reasoning_effort=_get("LLM_REASONING_EFFORT", ""),
        price_in_per_m=_get("PRICE_IN_PER_M", 0.0, float),
        price_out_per_m=_get("PRICE_OUT_PER_M", 0.0, float),
    )
    return replace(s, **overrides) if overrides else s
