from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.engine import Engine

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app_env(settings, monkeypatch):
    for k, v in {"PDF_DIR": settings.pdf_dir, "INDEX_DIR": settings.index_dir, "LOG_DIR": settings.log_dir,
                 "REPORT_DIR": settings.report_dir, "EMBED_BACKEND": "hash", "CHUNK_WORDS": 60, "CHUNK_OVERLAP": 10,
                 "MIN_DENSE_SCORE": "0.05", "NVIDIA_API_KEY": "", "WORKERS": 1}.items():
        monkeypatch.setenv(k, str(v))
    Engine(settings).sync()
    return settings


def test_app_chat_without_api_key_shows_evidence_and_logs(app_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("What was Quasar Corp total revenue in fiscal 2022?").run()
    assert not at.exception
    assert any("Answer generation is off" in i.value for i in at.info)
    from rag.metrics import read_log
    rows = read_log(app_env.log_dir)
    assert rows and rows[-1]["hits"][0]["doc"].startswith("Synthetic_2022")


@pytest.mark.parametrize("nav", ["📊 Analytics", "📚 Library"])
def test_other_pages_render(app_env, nav):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value(nav).run()
    assert not at.exception


def test_analytics_with_logged_queries_and_benchmark_report(app_env, tmp_path):
    import json

    from eval.benchmark import main as bench

    qf = tmp_path / "q.jsonl"
    qf.write_text("\n".join(json.dumps(x) for x in [
        {"id": "a", "question": "Quasar Corp revenue fiscal 2022", "relevant": [["Synthetic_2022_Annual_Report.pdf", 1]], "answerable": True},
        {"id": "b", "question": "airspeed of a swallow", "relevant": [], "answerable": False}]))
    assert bench(["--questions", str(qf), "--top-k", "3", "--no-rerank"]) == 0
    at = AppTest.from_file(APP, default_timeout=60).run()
    for q in ["revenue 2022", "dividend per share", "zorblax"]:
        at.chat_input[0].set_value(q).run()
    at.radio[0].set_value("📊 Analytics").run()
    assert not at.exception
    assert len(at.tabs) == 3
