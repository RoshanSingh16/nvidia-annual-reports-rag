"""Query log (JSONL) used by the analytics dashboard, plus retrieval-quality metrics."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_lock = threading.Lock()


def log_query(log_dir: Path, record: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with _lock, open(log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(log_dir: Path) -> list[dict]:
    p = log_dir / "queries.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---- ranking metrics (need human relevance labels) ----
def precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    top = list(ranked)[:k]
    return sum(1 for x in top if x in relevant) / k if k else 0.0


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len({x for x in list(ranked)[:k] if x in relevant}) / len(relevant)


def reciprocal_rank(ranked: Sequence, relevant: set) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(list(ranked)[:k], start=1) if x in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentiles(values: Iterable[float], min_n_p95: int = 20, min_n_p99: int = 100) -> dict:
    v = np.array(list(values), dtype=float)
    out = {"n": int(v.size), "median": float(np.median(v)) if v.size else None, "p95": None, "p99": None}
    if v.size >= min_n_p95:
        out["p95"] = float(np.percentile(v, 95))
    if v.size >= min_n_p99:
        out["p99"] = float(np.percentile(v, 99))
    return out
