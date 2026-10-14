"""Embedding backends. `fastembed` (ONNX, CPU) is the real one; `hash` is an offline test-only fallback."""
from __future__ import annotations

import logging
import re
import zlib
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Feature-hashing bag-of-words. NOT semantic; only for tests and offline smoke runs."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = zlib.crc32(tok.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # imported lazily: heavy

        log.info("Loading embedding model %s (first run downloads it)", model)
        self._model = TextEmbedding(model_name=model)
        self.name = model
        self._bge = "bge" in model.lower()
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._norm(np.array(list(self._model.embed(list(texts), batch_size=batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        q = (BGE_QUERY_PREFIX + text) if self._bge else text
        return self._norm(np.array(list(self._model.embed([q]))))[0]


def make_embedder(backend: str, model: str):
    if backend == "hash":
        return HashEmbedder()
    if backend == "fastembed":
        return FastEmbedder(model)
    raise ValueError(f"Unknown EMBED_BACKEND '{backend}' (use fastembed or hash)")
