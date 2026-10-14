"""CLI: python -m rag.cli {sync,coverage,stats,ask}"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .engine import Engine


def _bar(frac: float, msg: str) -> None:
    sys.stderr.write(f"\r[{'#' * int(frac * 30):<30}] {frac * 100:5.1f}%  {msg[:60]:<60}")
    sys.stderr.flush()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="index new/changed PDFs (incremental)")
    sub.add_parser("coverage", help="which files were ingested + year gaps")
    sub.add_parser("stats")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=None)
    a.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default=None)
    a.add_argument("--rerank", action="store_true")
    args = ap.parse_args(argv)

    eng = Engine()
    if args.cmd == "sync":
        rep = eng.sync(_bar)
        print()
        print(json.dumps({"processed": rep.processed, "reused": len(rep.reused), "removed": rep.removed,
                          "failed": rep.failed, "extract_s": round(rep.seconds_extract, 1),
                          "embed_s": round(rep.seconds_embed, 1), "total_s": round(rep.seconds_total, 1)}, indent=1))
    elif args.cmd == "stats":
        print(json.dumps(eng.store.stats(), indent=1, default=str))
    elif args.cmd == "coverage":
        docs = eng.store.manifest.get("docs", {})
        years = sorted({d["year"] for d in docs.values() if d.get("year")})
        print(f"{len(docs)} indexed documents")
        for name, d in sorted(docs.items(), key=lambda kv: (kv[1].get("year") or 0, kv[0])):
            warn = f"  !! {d['empty_pages']} pages without text" if d.get("empty_pages") else ""
            print(f"  {d.get('year')}  {d['doc_type']:<16} {name}  ({d['pages']} pages, {d['chunks']} chunks){warn}")
        if years:
            have = {(d["year"], d["doc_type"]) for d in docs.values()}
            missing = [y for y in range(1998, 2027) if (y, "annual report") not in have]
            print("\nYears 1998-2026 with no annual-report file detected (from file names):", missing or "none")
            print("Years with no proxy file detected:", [y for y in range(1998, 2027) if (y, "proxy statement") not in have] or "none")
        print("\nNote: type/year come from file names; files named without a year or 'annual'/'proxy' show as 'document'.")
    elif args.cmd == "ask":
        if not eng.ready:
            print("Index is empty. Put PDFs in data/pdfs and run: python -m rag.cli sync")
            return 1
        r = eng.retrieve(args.question, top_k=args.k, mode=args.mode, rerank=args.rerank or None)
        print(f"retrieval {r.total_ms:.1f} ms  {({k: round(v, 1) for k, v in r.timings_ms.items()})}")
        for h in r.hits:
            print(f"[{h.rank}] {h.doc} p.{h.page}  dense={h.dense:.3f} bm25={h.bm25:.1f}\n    {h.text[:160]!r}")
        if eng.llm.available and eng.evidence_ok(r):
            from .llm import Generation
            g = Generation()
            print("\n" + "".join(eng.answer_stream(args.question, r, gen=g)))
            print(f"\n[llm {g.latency_ms:.0f} ms, tokens in/out {g.prompt_tokens}/{g.completion_tokens}, cited {g.cited}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
