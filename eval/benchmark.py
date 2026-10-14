"""Reproducible retrieval benchmark.

Usage:
  python -m eval.benchmark --questions eval/questions.jsonl
  python -m eval.benchmark --questions eval/questions.jsonl --chunk-sizes 150:30,220:40,350:60 --top-k 3,6,10 --answers

Question file: JSONL, one object per line:
  {"id": "q1", "question": "...", "relevant": [["NVIDIA_2023_Annual_Report.pdf", 45]], "answerable": true,
   "answer_contains": ["26,974"]}
Metrics that need labels (P@K, R@K, MRR, nDCG@K) are computed ONLY for questions with a non-empty "relevant" list.
Unanswerable questions ("answerable": false) are used to calibrate the abstention threshold.
Nothing here is pre-filled: you must label questions yourself by opening the PDFs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np

from rag.config import load_settings
from rag.engine import Engine
from rag.llm import Generation
from rag.metrics import ndcg_at_k, percentiles, precision_at_k, recall_at_k, reciprocal_rank

log = logging.getLogger("benchmark")


def _rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    for q in qs:
        q["relevant"] = {(d, int(p)) for d, p in q.get("relevant", [])}
        q.setdefault("answerable", True)
    return qs


def unique_pages(hits) -> list[tuple[str, int]]:
    seen, out = set(), []
    for h in hits:
        k = (h.doc, h.page)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def run_config(eng: Engine, qs: list[dict], mode: str, rerank: bool, k: int, repeats: int = 3) -> dict:
    lat, rows = [], []
    for q in qs:
        # first call = cold (no cache) latency; extra repeats stabilise timing without hitting the result cache
        times = []
        for _ in range(repeats):
            r = eng.retrieve(q["question"], top_k=k, mode=mode, rerank=rerank, use_cache=False)
            times.append(r.total_ms)
        lat.append(float(np.median(times)))
        pages = unique_pages(r.hits)
        row = {"id": q.get("id"), "answerable": q["answerable"], "best_dense": r.best_dense}
        if q["relevant"] and q["answerable"]:
            row.update(p=precision_at_k(pages, q["relevant"], k), r=recall_at_k(pages, q["relevant"], k),
                       rr=reciprocal_rank(pages, q["relevant"]), ndcg=ndcg_at_k(pages, q["relevant"], k))
        rows.append(row)
    labelled = [r for r in rows if "p" in r]
    out = {"mode": mode, "rerank": rerank, "top_k": k, "n_questions": len(qs), "n_labelled": len(labelled),
           "latency_ms": percentiles(lat)}
    for key, name in [("p", "precision@k"), ("r", "recall@k"), ("rr", "mrr"), ("ndcg", "ndcg@k")]:
        out[name] = round(float(np.mean([r[key] for r in labelled])), 4) if labelled else None
    ans = [r["best_dense"] for r in rows if r["answerable"]]
    una = [r["best_dense"] for r in rows if not r["answerable"]]
    out["best_dense_answerable_mean"] = round(float(np.mean(ans)), 4) if ans else None
    out["best_dense_unanswerable_mean"] = round(float(np.mean(una)), 4) if una else None
    out["_rows"] = rows
    return out


def answer_quality(eng: Engine, qs: list[dict], k: int) -> dict | None:
    if not eng.llm.available:
        log.warning("No API key: skipping answer-quality evaluation")
        return None
    res = {"n": 0, "contains_ok": 0, "contains_total": 0, "abstain_correct": 0, "abstain_total": 0,
           "invalid_citations": 0, "answers_without_citation": 0, "llm_ms": [], "e2e_ms": [], "tokens_in": 0, "tokens_out": 0}
    for q in qs:
        t0 = time.perf_counter()
        r = eng.retrieve(q["question"], top_k=k, use_cache=False)
        g = Generation()
        try:
            "".join(eng.answer_stream(q["question"], r, gen=g))
        except Exception as exc:
            log.error("LLM failure on %s: %s", q.get("id"), exc)
            continue
        res["n"] += 1
        res["llm_ms"].append(g.latency_ms)
        res["e2e_ms"].append((time.perf_counter() - t0) * 1e3)
        res["tokens_in"] += g.prompt_tokens or 0
        res["tokens_out"] += g.completion_tokens or 0
        res["invalid_citations"] += len(g.invalid_cites)
        if q["answerable"]:
            if q.get("answer_contains"):
                res["contains_total"] += 1
                res["contains_ok"] += all(s.lower() in g.text.lower() for s in q["answer_contains"])
            res["answers_without_citation"] += (not g.cited and not g.abstained)
        res["abstain_total"] += 1
        res["abstain_correct"] += (g.abstained == (not q["answerable"]))
    res["llm_latency_ms"] = percentiles(res.pop("llm_ms"))
    res["end_to_end_ms"] = percentiles(res.pop("e2e_ms"))
    res["method"] = ("Deterministic checks only: required substrings present (answer_contains), abstention matches the "
                     "answerable flag, and citations reference real sources. Not an LLM judge; it does not measure paraphrased correctness.")
    return res


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--top-k", default="3,6,10")
    ap.add_argument("--chunk-sizes", default="", help="e.g. 150:30,220:40,350:60 (words:overlap); builds one index per setting")
    ap.add_argument("--answers", action="store_true", help="also run LLM answer checks (uses API tokens)")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args(argv)

    base = load_settings()
    qs = load_questions(args.questions)
    ks = [int(x) for x in args.top_k.split(",")]
    chunk_cfgs = [tuple(map(int, c.split(":"))) for c in args.chunk_sizes.split(",") if c] or [(base.chunk_words, base.chunk_overlap)]
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "embed_model": base.embed_model, "embed_backend": base.embed_backend,
              "questions_file": str(args.questions), "n_questions": len(qs),
              "n_labelled": sum(bool(q["relevant"]) and q["answerable"] for q in qs),
              "n_unanswerable": sum(not q["answerable"] for q in qs), "chunking": [], "configs": [], "answer_quality": None,
              "notes": ["Ranking metrics are page-level: a hit counts when its (pdf, page) is in the question's relevant list.",
                        "p95/p99 are reported only with >=20 / >=100 observations.",
                        "Baseline 'dense-no-cache' approximates a plain vector search; plug your original pipeline in for a true before/after."]}
    for cw, ov in chunk_cfgs:
        s = base.with_overrides(chunk_words=cw, chunk_overlap=ov, index_dir=base.index_dir)
        t0 = time.perf_counter()
        eng = Engine(s)
        rep = eng.sync()
        report["chunking"].append({"chunk_words": cw, "overlap": ov, "chunks": len(eng.store.chunks),
                                   "sync_s": round(time.perf_counter() - t0, 2), "extract_s": round(rep.seconds_extract, 2),
                                   "embed_s": round(rep.seconds_embed, 2), "reused_docs": len(rep.reused), "rss_mb": _rss_mb()})
        modes = [("dense", False), ("bm25", False), ("hybrid", False)] + ([] if args.no_rerank else [("dense", True), ("hybrid", True)])
        for (mode, rr), k in product(modes, ks):
            res = run_config(eng, qs, mode, rr, k)
            res.update(chunk_words=cw, overlap=ov)
            report["configs"].append(res)
            log.info("cw=%d ov=%d %-6s rerank=%-5s k=%-2d p50=%.1fms P@k=%s MRR=%s", cw, ov, mode, rr, k,
                     res["latency_ms"]["median"] or -1, res["precision@k"], res["mrr"])
        if args.answers and (cw, ov) == chunk_cfgs[0]:
            report["answer_quality"] = answer_quality(eng, qs, ks[len(ks) // 2])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base.report_dir.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in c.items() if k != "_rows"} for c in report["configs"]]
    for c in report["configs"]:
        c["_rows"] = c.pop("_rows")
    (base.report_dir / f"benchmark-{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    with open(base.report_dir / f"benchmark-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["chunk_words", "overlap", "mode", "rerank", "top_k", "n_labelled", "precision@k", "recall@k", "mrr", "ndcg@k"]
        w = csv.writer(f)
        w.writerow(cols + ["lat_median_ms", "lat_p95_ms", "lat_p99_ms", "lat_n"])
        for c in flat:
            L = c["latency_ms"]
            w.writerow([c.get(k) for k in cols] + [L["median"], L["p95"], L["p99"], L["n"]])
    print(f"Report written to {base.report_dir}/benchmark-{stamp}.json (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
