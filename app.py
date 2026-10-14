"""Streamlit app: Chat · Analytics · Library."""
from __future__ import annotations

import html
import io
import json
import logging
import time
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag.engine import Engine
from rag.llm import Generation, LLMError, LLMUnavailable
from rag.metrics import percentiles, read_log
from ui_theme import CSS, kpi_html, register_plotly_template

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="NVIDIA Financial Research Assistant", page_icon="🟩", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
register_plotly_template()

EXAMPLES = [
    "What was NVIDIA's total revenue in fiscal 2023, and how did it change from the prior year?",
    "Which risk factors relate to export controls on advanced chips?",
    "Summarise executive compensation highlights from the most recent proxy statement.",
    "How many employees did NVIDIA report in its latest annual report?",
]


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading index…")
def get_engine() -> Engine:
    return Engine()  # one engine/model instance per server process: no repeated model initialisation


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:  # older Streamlit
        st.dataframe(df, use_container_width=True, hide_index=True)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


def render_sources(sources: list[dict], n_context: int | None = None, cited: list[int] | None = None) -> None:
    cited = cited or []
    for s in sources:
        n = s["rank"]
        in_ctx = n_context is None or n <= n_context
        tag = chip(f"S{n}", "ok" if n in cited else "") + (chip("cited", "ok") if n in cited else "") + ("" if in_ctx else chip("not sent to LLM (context budget)", "warn"))
        st.markdown(
            f'<div class="src"><div class="t">{tag}{html.escape(s["doc"])} · page {s["page"]}</div>'
            f'<div class="m">{html.escape(str(s.get("doc_type", "")))} {s.get("year") or ""} · {s["kind"]} · '
            f'cosine {s["dense"]:.3f} · bm25 {s["bm25"]:.1f} · rank score {s["score"]:.4f}</div>'
            f'<pre>{html.escape(s["text"][:1400])}</pre></div>', unsafe_allow_html=True)


def sync_with_progress(eng: Engine) -> None:
    bar = st.progress(0.0, text="Starting…")
    t0 = time.perf_counter()

    def cb(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{msg}  ·  {time.perf_counter() - t0:.0f}s")

    try:
        rep = eng.sync(cb)
    except Exception as exc:
        bar.empty()
        st.error(f"Indexing failed: {exc}")
        logging.exception("sync failed")
        return
    bar.empty()
    if rep.failed:
        st.warning("Some PDFs could not be indexed:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in rep.failed.items()))
    st.success(f"Index ready in {rep.seconds_total:.1f}s · new/changed: {len(rep.processed)} · reused from cache: {len(rep.reused)} · removed: {len(rep.removed)}")


# ---------- sidebar ----------
eng = get_engine()
st.session_state.setdefault("messages", [])
with st.sidebar:
    st.markdown('<div class="brand"><span class="dot"></span>Filings Assistant</div><div class="subtle">NVIDIA annual reports &amp; proxy statements</div>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigate", [" Chat", " Analytics", " Library"], label_visibility="collapsed")
    st.divider()
    stats = eng.store.stats()
    st.markdown(chip(f"{stats['indexed_docs']} docs", "ok" if stats["indexed_docs"] else "warn") + chip(f"{stats['chunks']:,} chunks") +
                (chip("LLM ready", "ok") if eng.llm.available else chip("no API key", "warn")), unsafe_allow_html=True)
    with st.expander("Retrieval settings", expanded=False):
        top_k = st.slider("Top-k chunks", 2, 15, eng.s.top_k)
        mode = st.selectbox("Search mode", ["hybrid", "dense", "bm25"], index=["hybrid", "dense", "bm25"].index(eng.s.mode))
        rerank = st.toggle("Cross-encoder rerank", value=eng.s.rerank, help="More accurate, adds ~100-400 ms on CPU. Downloads a small model on first use.")
        detailed = st.toggle("Detailed answers", value=False)
        min_score = st.slider("Abstain below cosine", 0.0, 0.9, float(eng.s.min_dense_score), 0.01,
                              help="If the best retrieved chunk is below this similarity, no LLM call is made. Calibrate with the benchmark.")
        eng.s = eng.s.with_overrides(min_dense_score=min_score)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
# ---------- chat ----------
if page.endswith("Chat"):
    st.markdown(
        '<div style="padding-top: 12px;">'
        '<p class="page-title">Ask the filings</p>'
        '<p class="subtle" title="Answers are generated only from retrieved PDF passages and cite document and page." '
        'style="cursor:help;">'
        'Answers are generated only from retrieved PDF passages and cite document and page.'
        '</p></div>',
        unsafe_allow_html=True
    )
    if not eng.ready:
        n_pdfs = len(eng.store.list_pdfs())
        if n_pdfs == 0:
            st.markdown('<div class="callout"><b>No PDFs found.</b> Copy your PDFs into <code>data/pdfs/</code> and press the button below. '
                        'The first index takes a few minutes; later starts load from disk instantly.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="callout"><b>{n_pdfs} PDFs found, not indexed yet.</b></div>', unsafe_allow_html=True)
        if st.button("Build index", type="primary"):
            sync_with_progress(eng)
            st.rerun()
        st.stop()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                meta = m["meta"]
                st.markdown("".join(chip(x, k) for x, k in meta["chips"]), unsafe_allow_html=True)
                with st.expander(f"Evidence · {len(m['sources'])} retrieved chunks"):
                    render_sources(m["sources"], meta.get("n_context"), meta.get("cited"))

    if not st.session_state.messages:
        st.markdown("**Try one of these**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}"):
                st.session_state.pending = ex
                st.rerun()

    q = st.chat_input("Ask about revenue, risk factors, governance, headcount…") or st.session_state.pop("pending", None)
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            gen = Generation()
            error = None
            with st.spinner("Searching documents…"):
                r = eng.retrieve(q, top_k=top_k, mode=mode, rerank=rerank)
            sources = [dict(rank=h.rank, doc=h.doc, page=h.page, year=h.year, doc_type=h.doc_type, kind=h.kind,
                            text=h.text, dense=h.dense, bm25=h.bm25, score=h.score) for h in r.hits]
            history = [m for m in st.session_state.messages[:-1] if m["role"] == "user"]
            text = ""
            try:
                if not eng.llm.available and eng.evidence_ok(r):
                    raise LLMUnavailable("Answer generation is off: add NVIDIA_API_KEY to .env and restart. The retrieved evidence is shown below.")
                text = st.write_stream(eng.answer_stream(q, r, detailed, history, gen))
            except LLMUnavailable as exc:
                error = str(exc)
                st.info(error)
            except LLMError as exc:
                error = str(exc)
                st.error(error + " The retrieved evidence is shown below.")
            except Exception as exc:  # never crash the chat
                logging.exception("answer failed")
                error = f"{type(exc).__name__}: {exc}"
                st.error("Something went wrong while generating the answer. Retrieved evidence is shown below.")
            if gen.invalid_cites:
                st.warning(f"The model referenced source tags that do not exist ({gen.invalid_cites}); ignore those citations.")
            chips = [(f"retrieval {r.total_ms:.0f} ms" + (" · cached" if r.cache_hit else ""), "ok")]
            if gen.latency_ms:
                chips.append((f"generation {gen.latency_ms / 1000:.1f} s", ""))
            if gen.prompt_tokens is not None:
                chips.append((f"{gen.prompt_tokens}+{gen.completion_tokens or 0} tokens", ""))
            if gen.cost_usd is not None:
                chips.append((f"≈ ${gen.cost_usd:.4f}", ""))
            if gen.abstained:
                chips.append(("insufficient evidence", "warn"))
            st.markdown("".join(chip(x, k) for x, k in chips), unsafe_allow_html=True)
            with st.expander(f"Evidence · {len(sources)} retrieved chunks", expanded=bool(gen.cited)):
                if sources:
                    render_sources(sources, gen.context_sources or None, gen.cited)
                else:
                    st.write("Nothing was retrieved.")
            eng.record(q, r, gen if gen.latency_ms or gen.abstained else None, error)
        st.session_state.messages.append({"role": "assistant", "content": text or error or "", "sources": sources,
                        "meta": {"chips": chips, "n_context": gen.context_sources or None, "cited": gen.cited}})

# ---------- analytics ----------
elif page.endswith("Analytics"):
    st.markdown('<p class="page-title">Analytics</p><p class="subtle">Measured values come from this app\'s own query log and index manifest. Quality metrics appear only after you run the labelled benchmark.</p>', unsafe_allow_html=True)
    stats = eng.store.stats()
    ls = stats.get("last_sync") or {}
    st.markdown(kpi_html([
        ("PDFs on disk", f"{stats['pdfs_on_disk']}", f"{stats['indexed_docs']} indexed"),
        ("Pages", f"{stats['pages']:,}", f"{stats['empty_pages']} without text"),
        ("Chunks", f"{stats['chunks']:,}", f"{stats['table_chunks']:,} table chunks"),
        ("Vector memory", f"{stats['index_mb']} MB", "float32 embeddings"),
        ("Last sync", f"{ls.get('seconds_total', 0):.1f}s" if ls else "—", f"extract {ls.get('seconds_extract', 0):.0f}s · embed {ls.get('seconds_embed', 0):.0f}s" if ls else "no sync yet"),
    ]), unsafe_allow_html=True)

    rows = read_log(eng.s.log_dir)
    tab_live, tab_docs, tab_bench = st.tabs(["Live performance", "Sources & relevance", "Benchmark & quality"])

    if rows:
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        df["qn"] = range(1, len(df) + 1)
        df["llm_ms"] = pd.to_numeric(df["llm_ms"], errors="coerce")
        df["retrieval_ms"] = pd.to_numeric(df["retrieval_ms"], errors="coerce")
    with tab_live:
        if not rows:
            st.markdown('<div class="callout"><b>No queries logged yet.</b> Ask a question in Chat and charts will appear here.</div>', unsafe_allow_html=True)
        else:
            modes = st.multiselect("Search mode", sorted(df["mode"].unique()), default=sorted(df["mode"].unique()))
            d = df[df["mode"].isin(modes)]
            rp, lp = percentiles(d["retrieval_ms"].dropna()), percentiles(d["llm_ms"].dropna())
            fmt = lambda v: "—" if v is None else f"{v:.0f} ms"
            st.markdown(kpi_html([
                ("Queries", f"{len(d)}", f"{int(d['cache_hit'].sum())} cache hits"),
                ("Retrieval median", fmt(rp["median"]), f"p95 {fmt(rp['p95'])} (needs ≥20 obs)"),
                ("Generation median", fmt(lp["median"]), f"p95 {fmt(lp['p95'])} · n={lp['n']}"),
                ("Abstained", f"{int(d['abstained'].fillna(False).sum())}", "insufficient evidence"),
                ("Tokens", f"{int(d['prompt_tokens'].fillna(0).sum() + d['completion_tokens'].fillna(0).sum()):,}", "measured from API usage"),
            ]), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            fig = go.Figure()
            fig.add_bar(x=d["qn"], y=d["retrieval_ms"], name="Retrieval (ms)", marker_color="#76B900", hovertemplate="Q%{x}<br>retrieval %{y:.1f} ms<extra></extra>")
            fig.add_bar(x=d["qn"], y=d["llm_ms"], name="LLM generation (ms)", marker_color="#4DA3FF", hovertemplate="Q%{x}<br>LLM %{y:.0f} ms<extra></extra>")
            fig.update_layout(barmode="stack", title="Latency per query (retrieval vs generation)", xaxis_title="Query #", yaxis_title="ms")
            c1.plotly_chart(fig)
            steps = pd.DataFrame([t for t in d["timings_ms"] if isinstance(t, dict)])
            if not steps.empty:
                steps = steps.drop(columns=[c for c in ["total", "cache"] if c in steps], errors="ignore").mean().reset_index()
                steps.columns = ["step", "ms"]
                c2.plotly_chart(px.bar(steps, x="step", y="ms", title="Mean retrieval time by step (uncached)", text_auto=".2f"))
            c3, c4 = st.columns(2)
            vol = d.set_index("time").resample("1h" if (d["time"].max() - d["time"].min()).days < 3 else "1D").size().reset_index(name="queries")
            c3.plotly_chart(px.bar(vol, x="time", y="queries", title="Query volume"))
            tr = go.Figure()
            tr.add_scatter(x=d["time"], y=d["retrieval_ms"], mode="lines+markers", name="Retrieval")
            tr.add_scatter(x=d["time"], y=d["llm_ms"], mode="lines+markers", name="Generation")
            tr.update_layout(title="Response-time trend", yaxis_title="ms", yaxis_type="log")
            c4.plotly_chart(tr)
            if d["cost_usd"].notna().any():
                st.plotly_chart(px.line(d.assign(cum=d["cost_usd"].fillna(0).cumsum()), x="time", y="cum", title="Estimated cumulative API cost (USD)"))
            st.download_button("Download query log (CSV)", d.drop(columns=["hits", "timings_ms", "qn"]).to_csv(index=False), "query_log.csv", "text/csv")

    with tab_docs:
        if not rows:
            st.markdown('<div class="callout">Retrieval relevance charts need logged queries.</div>', unsafe_allow_html=True)
        else:
            hits = pd.DataFrame([{**h, "qn": i + 1, "mode": r["mode"], "rank": j + 1} for i, r in enumerate(rows) for j, h in enumerate(r["hits"])])
            if hits.empty:
                st.info("No hits logged yet.")
            else:
                st.caption("Cosine/BM25 values are similarity scores, **not** correctness. They indicate how close a chunk is to the query; only labelled benchmarks measure relevance.")
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(hits, x="dense", nbins=30, title="Retrieved-chunk cosine score distribution", labels={"dense": "cosine similarity"}))
                top = hits["doc"].value_counts().head(12).reset_index()
                top.columns = ["doc", "times retrieved"]
                c2.plotly_chart(px.bar(top.iloc[::-1], x="times retrieved", y="doc", orientation="h", title="Most retrieved documents"))
                sel = st.selectbox("Page distribution for document", ["(all)"] + sorted(hits["doc"].unique()))
                hp = hits if sel == "(all)" else hits[hits["doc"] == sel]
                st.plotly_chart(px.histogram(hp, x="page", nbins=60, color="doc" if sel == "(all)" else None, title="Retrieved page distribution", labels={"page": "page number"}))
                kdf = hits.groupby("rank")["dense"].agg(["mean", "max", "min"]).reset_index()
                kf = go.Figure()
                kf.add_scatter(x=kdf["rank"], y=kdf["mean"], mode="lines+markers", name="mean cosine")
                kf.add_scatter(x=kdf["rank"], y=kdf["max"], mode="lines", name="max", line=dict(dash="dot"))
                kf.add_scatter(x=kdf["rank"], y=kdf["min"], mode="lines", name="min", line=dict(dash="dot"))
                kf.update_layout(title="Score by rank position (how quickly relevance decays with k)", xaxis_title="rank", yaxis_title="cosine")
                st.plotly_chart(kf)

    with tab_bench:
        reports = sorted(eng.s.report_dir.glob("benchmark-*.json")) if eng.s.report_dir.exists() else []
        if not reports:
            st.markdown('<div class="callout"><b>No benchmark report yet.</b> Precision@K, Recall@K, MRR and nDCG@K require <b>human relevance labels</b> '
                        '(which PDF page answers each question). Copy <code>eval/questions.example.jsonl</code> to <code>eval/questions.jsonl</code>, label ~30-50 questions, then run '
                        '<code>python -m eval.benchmark --questions eval/questions.jsonl</code>. Faithfulness / answer correctness need reference answers and an LLM key; the benchmark reports only the deterministic checks it can actually verify.</div>', unsafe_allow_html=True)
        else:
            pick = st.selectbox("Report", [p.name for p in reports][::-1])
            rep = json.loads((eng.s.report_dir / pick).read_text())
            st.markdown(kpi_html([("Questions", str(rep["n_questions"]), f"{rep['n_labelled']} labelled · {rep['n_unanswerable']} unanswerable"),
                                    ("Embedding", rep["embed_model"].split("/")[-1], rep["embed_backend"]),
                                    ("Created", rep["created"], "measured")]), unsafe_allow_html=True)
            cfg = pd.DataFrame([{k: v for k, v in c.items() if k not in {"_rows", "latency_ms"}} | {"lat_median_ms": c["latency_ms"]["median"], "lat_p95_ms": c["latency_ms"]["p95"]} for c in rep["configs"]])
            cfg["strategy"] = cfg["mode"] + cfg["rerank"].map({True: "+rerank", False: ""})
            f1, f2 = st.columns(2)
            cw = f1.selectbox("Chunk size / overlap", sorted({(a, b) for a, b in zip(cfg.chunk_words, cfg.overlap)}), format_func=lambda t: f"{t[0]} words / {t[1]} overlap")
            metric = f2.selectbox("Metric", ["mrr", "precision@k", "recall@k", "ndcg@k"])
            sub = cfg[(cfg.chunk_words == cw[0]) & (cfg.overlap == cw[1])]
            if sub[metric].notna().any():
                st.plotly_chart(px.bar(sub, x="top_k", y=metric, color="strategy", barmode="group", title=f"{metric} by strategy and top-k"))
            else:
                st.info("This report has no labelled questions, so ranking metrics are not available.")
            st.plotly_chart(px.bar(sub, x="top_k", y="lat_median_ms", color="strategy", barmode="group", title="Median retrieval latency (ms)"))
            if len(rep["chunking"]) > 1:
                ch = pd.DataFrame(rep["chunking"])
                st.plotly_chart(px.bar(ch, x=ch.chunk_words.astype(str) + "/" + ch.overlap.astype(str), y="sync_s", title="Indexing time by chunk setting (s)", labels={"x": "words/overlap"}))
            show_table(cfg.drop(columns=["strategy"]))
            if rep.get("answer_quality"):
                st.subheader("Answer checks")
                aq = rep["answer_quality"]
                st.caption(aq["method"])
                st.json({k: v for k, v in aq.items() if k != "method"})
            st.download_button("Download report (JSON)", json.dumps(rep, indent=1), pick, "application/json")
            csvp = (eng.s.report_dir / pick).with_suffix(".csv")
            if csvp.exists():
                st.download_button("Download report (CSV)", csvp.read_text(), csvp.name, "text/csv")

# ---------- library ----------
else:
    st.markdown('<p class="page-title">Library</p><p class="subtle">Every file the system ingests, and which years are covered.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    if c1.button("Sync index now", type="primary"):
        sync_with_progress(eng)
        st.rerun()
    c2.caption("Only new, modified or removed PDFs are re-processed; unchanged files load from the on-disk cache.")
    docs = eng.store.manifest.get("docs", {})
    if not docs:
        st.markdown('<div class="callout">Nothing indexed yet. Put PDFs in <code>data/pdfs/</code> and click <b>Sync index now</b>.</div>', unsafe_allow_html=True)
    else:
        df = pd.DataFrame([{"file": n, "year": d.get("year"), "type": d.get("doc_type"), "pages": d.get("pages"), "chunks": d.get("chunks"),
                            "table chunks": d.get("table_chunks"), "pages w/o text": d.get("empty_pages")} for n, d in docs.items()]).sort_values(["year", "file"])
        if df["pages w/o text"].sum():
            st.warning(f"{int(df['pages w/o text'].sum())} pages have no extractable text (likely scanned images). Set OCR=1 in .env (requires Tesseract) and re-sync to include them.")
        years = list(range(1998, 2027))
        types = ["annual report", "proxy statement", "document"]
        have = Counter((d.get("year"), d.get("doc_type")) for d in docs.values())
        z = [[have.get((y, t), 0) for y in years] for t in types]
        hm = go.Figure(go.Heatmap(z=z, x=years, y=types, colorscale=[[0, "#121820"], [1, "#76B900"]], showscale=False,
                                hovertemplate="%{y} · %{x}: %{z} file(s)<extra></extra>", xgap=3, ygap=3))
        hm.update_layout(title="Coverage by year (derived from file names; dark = no file detected)", height=240, xaxis=dict(dtick=2))
        st.plotly_chart(hm)
        show_table(df)
