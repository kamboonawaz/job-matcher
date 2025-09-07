from __future__ import annotations
import os

from src.pipeline.build_index import build_and_save
from src.index.vectorstore import TfidfJobIndex
from src.utils.config import load_config
from src.pipeline.report_matches import score_resumes


def test_build_and_query():
    cfg = load_config()
    build_and_save()

    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg.paths.artifacts_dir)
    idx = TfidfJobIndex.load(artifacts_dir)
    res = idx.query("python machine learning aws", top_k=3)
    assert len(res) > 0
    assert all("job_id" in r and "score" in r for r in res)


def test_various_queries():
    cfg = load_config()
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg.paths.artifacts_dir)
    idx = TfidfJobIndex.load(artifacts_dir)

    # Resume-driven queries (ensure no crashes and some results)
    for text in [
        "Kubernetes Docker Terraform MLflow",
        "Transformers RAG embeddings FAISS LangChain",
        "SQL A/B testing statistics time series",
    ]:
        res = idx.query(text, top_k=5)
        assert len(res) > 0


def test_reporting_outputs():
    rows_csv, rows_md = score_resumes(top_k=3)
    # Ensure at least one resume x job row generated
    assert len(rows_csv) > 0 and len(rows_md) > 0
