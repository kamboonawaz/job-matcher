from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass
from typing import List

from src.index.vectorstore import TfidfJobIndex, CorpusItem
from src.utils.config import load_config


@dataclass
class JobRecord:
    job_id: str
    title: str
    company: str
    location: str
    description: str


def read_jobs_csv(path: str) -> List[JobRecord]:
    df = pd.read_csv(path)
    needed = ["job_id", "title", "company", "location", "description"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in jobs CSV")
    jobs: List[JobRecord] = [JobRecord(**row._asdict()) if hasattr(row, "_asdict") else JobRecord(**row) for row in df.to_dict(orient="records")]
    return jobs


def build_and_save():
    cfg = load_config()
    jobs = read_jobs_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.jobs_csv))

    docs = [
        f"{j.title} {j.company} {j.location} {j.description}" for j in jobs
    ]
    meta = [CorpusItem(job_id=j.job_id, title=j.title, company=j.company, location=j.location, description=j.description) for j in jobs]
    idx = TfidfJobIndex.from_corpus(
        docs,
        meta,
        max_features=cfg.vectorizer.max_features,
        ngram_range=cfg.vectorizer.ngram_range,
        min_df=cfg.vectorizer.min_df,
        max_df=cfg.vectorizer.max_df,
    )

    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.artifacts_dir)
    idx.save(artifacts_dir)
    print(f"Saved artifacts to {artifacts_dir}")


if __name__ == "__main__":
    build_and_save()
