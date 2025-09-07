from __future__ import annotations
import os
import glob
import csv
from typing import List, Tuple

from src.index.vectorstore import TfidfJobIndex
from src.utils.config import load_config


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def score_resumes(top_k: int = 5) -> Tuple[List[dict], List[dict]]:
    cfg = load_config()
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.artifacts_dir)
    resumes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.resumes_dir)

    idx = TfidfJobIndex.load(artifacts_dir)

    rows_csv: List[dict] = []
    rows_md: List[dict] = []

    for resume_path in sorted(glob.glob(os.path.join(resumes_dir, "*.txt"))):
        resume_id = os.path.splitext(os.path.basename(resume_path))[0]
        text = load_text(resume_path)
        results = idx.query(text, top_k=top_k)
        for rank, r in enumerate(results, start=1):
            row = {
                "resume_id": resume_id,
                "rank": rank,
                "job_id": r["job_id"],
                "title": r["title"],
                "company": r["company"],
                "location": r["location"],
                "score": f"{r['score']:.4f}",
            }
            rows_csv.append(row)
            rows_md.append(row)
    return rows_csv, rows_md


def write_reports(rows_csv: List[dict], rows_md: List[dict]):
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    # CSV
    csv_path = os.path.join(out_dir, "resume_job_matches.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["resume_id", "rank", "job_id", "title", "company", "location", "score"])
        w.writeheader()
        w.writerows(rows_csv)
    # Markdown
    md_path = os.path.join(out_dir, "resume_job_matches.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Resume to Job Top Matches\n\n")
        current = None
        for row in rows_md:
            if current != row["resume_id"]:
                current = row["resume_id"]
                f.write(f"\n## {current}\n\n")
                f.write("| Rank | Job ID | Title | Company | Location | Score |\n")
                f.write("|---:|---:|---|---|---|---:|\n")
            f.write(f"| {row['rank']} | {row['job_id']} | {row['title']} | {row['company']} | {row['location']} | {row['score']} |\n")

    print(f"Wrote reports to: {csv_path} and {md_path}")


if __name__ == "__main__":
    rows_csv, rows_md = score_resumes(top_k=5)
    write_reports(rows_csv, rows_md)
