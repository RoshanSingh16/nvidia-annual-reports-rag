import math
import shutil

from rag.engine import Engine
from rag.ingest import split_text
from rag.llm import build_context, validate_citations
from rag.metrics import ndcg_at_k, percentiles, precision_at_k, recall_at_k, reciprocal_rank
from tests.conftest import make_pdf


def test_chunks_have_exact_pages_and_overlap():
    text = "\n".join(f"line {i} " + "w " * 9 for i in range(40))
    chunks = split_text(text, size=50, overlap=12)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 50 + 12 for c in chunks)
    # consecutive chunks share overlap text
    assert set(chunks[0].splitlines()) & set(chunks[1].splitlines())


def test_sync_is_incremental_and_persistent(settings):
    eng = Engine(settings)
    r1 = eng.sync()
    assert len(r1.processed) == 3 and not r1.failed
    assert eng.store.stats()["chunks"] > 0
    # restart: nothing re-processed, index loads from disk with no PDF parsing
    eng2 = Engine(settings)
    assert eng2.ready
    r2 = eng2.sync()
    assert r2.processed == [] and len(r2.reused) == 3
    # new PDF -> only that one is processed
    make_pdf(settings.pdf_dir / "Synthetic_2023_Annual_Report.pdf", ["Quasar Corp revenue 9,000 million USD fiscal 2023."], 12)
    r3 = eng2.sync()
    assert r3.processed == ["Synthetic_2023_Annual_Report.pdf"]
    # modified PDF -> reprocessed
    make_pdf(settings.pdf_dir / "Synthetic_2023_Annual_Report.pdf", ["Quasar Corp revenue 9,100 million USD fiscal 2023."], 12)
    assert eng2.sync().processed == ["Synthetic_2023_Annual_Report.pdf"]
    # deleted PDF -> removed from active index
    (settings.pdf_dir / "Synthetic_2023_Annual_Report.pdf").unlink()
    assert eng2.sync().removed == ["Synthetic_2023_Annual_Report.pdf"]
    assert all(c["doc"] != "Synthetic_2023_Annual_Report.pdf" for c in eng2.store.chunks)


def test_retrieval_returns_correct_page_and_years(settings):
    eng = Engine(settings)
    eng.sync()
    r = eng.retrieve("What was Quasar Corp total revenue in fiscal 2022?", top_k=3)
    assert r.hits[0].doc == "Synthetic_2022_Annual_Report.pdf" and r.hits[0].page == 1
    r = eng.retrieve("zorblax supply constraints", top_k=3, mode="bm25")
    assert (r.hits[0].doc, r.hits[0].page) == ("Synthetic_2021_Annual_Report.pdf", 2)
    # multi-year question covers both years
    r = eng.retrieve("Compare Quasar revenue 2021 and 2022", top_k=4)
    assert {h.year for h in r.hits} >= {2021, 2022}
    assert {"embed", "dense", "bm25", "fuse", "total"} <= set(r.timings_ms)


def test_query_cache(settings):
    eng = Engine(settings)
    eng.sync()
    a = eng.retrieve("quarterly dividend per share")
    b = eng.retrieve("quarterly dividend per share")
    assert not a.cache_hit and b.cache_hit and b.total_ms < a.total_ms


def test_abstains_without_llm_call_when_evidence_weak(settings):
    eng = Engine(settings.with_overrides(min_dense_score=0.99))
    eng.sync()
    r = eng.retrieve("what is the airspeed velocity of an unladen swallow")
    out = "".join(eng.answer_stream("q", r))
    assert "can't answer" in out


def test_citation_validation_never_accepts_invented_sources():
    assert validate_citations("Revenue rose [S1][S3] and also [S9].", 3) == ([1, 3], [9])
    assert validate_citations("no cites", 3) == ([], [])


def test_context_budget(settings):
    eng = Engine(settings)
    eng.sync()
    r = eng.retrieve("revenue", top_k=6)
    ctx, n = build_context(r.hits, 400)
    assert len(ctx) <= 400 + 200 and n >= 1


def test_ranking_metrics_hand_computed():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert reciprocal_rank(ranked, rel) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert abs(ndcg_at_k(ranked, rel, 4) - expected) < 1e-9
    assert percentiles([1, 2, 3])["p95"] is None  # not enough observations -> not reported


def test_corrupt_pdf_does_not_break_sync(settings):
    (settings.pdf_dir / "broken_2020.pdf").write_bytes(b"not a pdf")
    eng = Engine(settings)
    rep = eng.sync()
    assert "broken_2020.pdf" in rep.failed and len(rep.processed) == 3


def test_sparse_bm25_matches_reference_formula():
    import math

    from rag.bm25 import SparseBM25

    docs = [["a", "b", "b"], ["b", "c"], ["c", "c", "d", "e"]]
    bm = SparseBM25.build(docs)
    n, k1, b = 3, 1.5, 0.75
    avg = sum(map(len, docs)) / n
    idf_b = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    want = idf_b * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / avg))
    got = bm.get_scores(["b"])
    assert abs(got[0] - want) < 1e-4 and got[2] == 0
    assert bm.get_scores(["zzz"]).sum() == 0
