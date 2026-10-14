"""Sparse-matrix BM25 (Okapi). Query time is a few column sums instead of a Python loop over the corpus."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp


class SparseBM25:
    def __init__(self, matrix: sp.csc_matrix, vocab: dict[str, int]):
        self.matrix, self.vocab = matrix, vocab

    @classmethod
    def build(cls, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> "SparseBM25":
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        dl = np.zeros(len(docs), dtype=np.float32)
        for i, toks in enumerate(docs):
            dl[i] = len(toks)
            for t, c in Counter(toks).items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(i), cols.append(j), tfs.append(c)
        n, v = len(docs), len(vocab)
        tf = sp.coo_matrix((np.array(tfs, np.float32), (rows, cols)), shape=(n, v)).tocsr()
        df = np.bincount(tf.indices, minlength=v).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = max(float(dl.mean()), 1e-9) if n else 1.0
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = tf.data * (k1 + 1) / denom * idf[tf.col]
        return cls(sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=(n, v)), vocab)

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        ids = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, ids].sum(axis=1)).ravel()

    def save(self, stem: Path) -> None:
        sp.save_npz(str(stem) + ".npz", self.matrix)
        Path(str(stem) + ".vocab.json").write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, stem: Path) -> "SparseBM25":
        m = sp.load_npz(str(stem) + ".npz").tocsc()
        return cls(m, json.loads(Path(str(stem) + ".vocab.json").read_text()))
