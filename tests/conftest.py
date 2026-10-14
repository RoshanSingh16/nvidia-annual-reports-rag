import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import load_settings  # noqa: E402

# Synthetic content ONLY for tests. These are NOT NVIDIA facts.
DOCS = {
    "Synthetic_2021_Annual_Report.pdf": [
        "Quasar Corp total revenue was 5,000 million USD in fiscal 2021. Gamma segment grew strongly.",
        "Risk factors: zorblax supply constraints may affect manufacturing capacity.",
    ],
    "Synthetic_2022_Annual_Report.pdf": [
        "Quasar Corp total revenue was 7,500 million USD in fiscal 2022. Headcount reached 1,200 employees.",
        "The board approved a quarterly dividend of 0.04 USD per share.",
    ],
    "Synthetic_2022_Proxy_Statement.pdf": [
        "Proxy: the annual meeting of stockholders will elect nine directors. Executive compensation is described below.",
    ],
}


def make_pdf(path: Path, pages: list[str], filler: int = 0):
    doc = pymupdf.open()
    for text in pages:
        pg = doc.new_page()
        body = text + "\n" + "\n".join(f"Filler line {i} about general corporate matters and notes." for i in range(filler))
        pg.insert_text((50, 60), body, fontsize=9)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def settings(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for name, pages in DOCS.items():
        make_pdf(pdfs / name, pages, filler=12)
    return load_settings(pdf_dir=pdfs, index_dir=tmp_path / "index", log_dir=tmp_path / "logs",
                         report_dir=tmp_path / "reports", embed_backend="hash", workers=2,
                         chunk_words=60, chunk_overlap=10, min_dense_score=0.05)
