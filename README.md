# NVIDIA Filings RAG Chatbot

Question answering over NVIDIA annual reports and proxy statements (1998-2026) with **page-level citations**,
persistent incremental indexing, hybrid retrieval, abstention when evidence is weak, an analytics dashboard and a
reproducible retrieval benchmark.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # Windows: copy .env.example .env  -> then set NVIDIA_API_KEY
# put your PDFs in data/pdfs/  (e.g. NVIDIA_2023_Annual_Report.pdf, NVIDIA_2023_Proxy_Statement.pdf)
python -m rag.cli sync        # first index; later runs only process new/changed/removed PDFs
python -m rag.cli coverage    # which files were ingested + years with no file detected
streamlit run app.py
```
* The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB, CPU/ONNX, no GPU or torch needed).
* Without `NVIDIA_API_KEY` the app still retrieves and shows evidence; it just doesn't generate answers.
* Windows / multiprocessing trouble: set `WORKERS=1` in `.env`.
* The LLM endpoint is any OpenAI-compatible API. Default: NVIDIA `https://integrate.api.nvidia.com/v1`, model `meta/llama-3.1-70b-instruct` (change `LLM_MODEL` to one available to your key).

## Architecture
```
PDFs -> PyMuPDF (text + number-dense-page tables as Markdown) -> page-bounded chunks (220 words / 40 overlap)
     -> bge-small embeddings (batched, cached per PDF by SHA-256 + config)  -> FAISS (persisted)  + sparse BM25 (persisted)
query -> embed (LRU cached) -> dense top-N + BM25 top-N -> RRF fusion (+ soft year boost) -> optional cross-encoder rerank
      -> dedupe -> top-k -> evidence gate (abstain if best cosine < MIN_DENSE_SCORE) -> LLM with numbered sources -> citation validation
```
| File | Role |
|---|---|
| `rag/ingest.py` | PDF parsing, table->Markdown, chunking (chunks never cross pages, so citations are exact) |
| `rag/store.py` | Persistent per-PDF cache, incremental sync, FAISS + BM25 persistence, manifest/stats |
| `rag/bm25.py` | Sparse-matrix BM25 (~4 ms/query at 28k chunks vs ~85 ms for a pure-Python implementation) |
| `rag/retrieve.py` | Hybrid search, RRF, rerank, dedupe, multi-year coverage, caches, per-stage timings |
| `rag/llm.py` | Grounded prompt, streaming, retries/timeouts, token + cost tracking, citation validation |
| `rag/engine.py` | Facade: sync, retrieve, answer, query logging |
| `app.py`, `ui_theme.py` | Streamlit UI (Chat / Analytics / Library), Inter + Plus Jakarta Sans, Plotly theme |
| `eval/benchmark.py` | Benchmark: strategies x top-k x chunk sizes, P@K/R@K/MRR/nDCG@K, latency percentiles, answer checks |

## Design decisions
* **Citations come from metadata, not from the model.** Each source is `file + page` from the index. The model may only cite `[S#]` tags it was given; tags outside range are flagged and ignored.
* **Abstention in two layers:** a similarity gate before any LLM call (`MIN_DENSE_SCORE`, *uncalibrated default - calibrate it with your labelled questions*), then a prompt rule to answer "I can't answer this from the retrieved documents."
* **Numbers:** number-dense pages are also parsed with PyMuPDF's table finder and stored as Markdown table chunks (header repeated on split); the prompt demands metric name, value, unit and reporting period. Table extraction is heuristic and can miss/garble complex financial tables - verify against the evidence panel.
* **Years:** years in the question softly boost matching documents (never a hard filter, because fiscal-year naming differs from file-name years) and multi-year questions are guaranteed chunks from each named year when available.
* **Persistence:** per-PDF cache keyed by `sha256 + embedding/chunk config`. Restart = load from disk; new/modified PDF = only that file is re-processed; deleted PDF = dropped from the active index.
* **Scale:** exact flat FAISS (fine to ~100k chunks). `INDEX_TYPE=hnsw` is available but untested at scale.

## Evaluation
1. Unit/integration tests: `pytest -q`
2. Label questions yourself (open the PDFs, note the page that contains the answer): copy `eval/questions.example.jsonl` -> `eval/questions.jsonl`.
3. `python -m eval.benchmark --questions eval/questions.jsonl --top-k 3,6,10 --chunk-sizes 150:30,220:40,350:60 [--answers]`
   Outputs `data/reports/benchmark-*.json|csv`, viewable (with filters and downloads) in the dashboard.

Metrics and when they are valid:
| Metric | Needs |
|---|---|
| Precision@K, Recall@K, MRR, nDCG@K | human labels: `relevant` (pdf, page) per question; page-level |
| Latency median / p95 / p99 | nothing; p95 needs >=20 and p99 >=100 observations or they are not reported |
| Answer checks (`--answers`) | API key; deterministic only: required substrings (`answer_contains`), abstention correctness, citation validity |
| Faithfulness / answer relevance | **not implemented** - would need an LLM judge or human review; no numbers are produced for them |

No benchmark result is shipped. The "existing vs optimized" comparison needs your original pipeline as a baseline; the
benchmark's `dense` row is a plain vector-search stand-in only.

## Inputs and example questions
Natural-language questions; optional sidebar controls for top-k, search mode, rerank, detail level, abstention threshold.
* "What was NVIDIA's total revenue in fiscal 2023 versus fiscal 2022?"
* "Which risk factors mention export controls?"
* "What were the main items on the agenda of the 2022 annual meeting?"
* Unanswerable (should abstain): "What is AMD's 2024 data-center revenue?"

## Known limitations
* Verified only on synthetic PDFs in a sandbox without Hugging Face / NVIDIA API access (see test notes in the hand-off). Real embedding quality, rerank, LLM answers and real-filing table extraction are untested.
* Scanned PDFs: pages without a text layer are counted and flagged; set `OCR=1` (needs Tesseract) to OCR them.
* Year/type are inferred from file names; coverage gaps shown in the Library page are only as good as the naming.
* Text order uses PyMuPDF's native order (`SORT_TEXT=0`) for speed; multi-column proxies may read better with `SORT_TEXT=1` (slower).
* Conversation memory is minimal (previous user questions only); follow-ups that rely on earlier answers may retrieve poorly.
