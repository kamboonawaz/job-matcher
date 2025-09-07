# Job Matcher — DS/AI portfolio project

Production‑ready baseline of a job recommendation service that ranks job postings for a candidate. Starts with TF‑IDF + cosine, designed to evolve into embeddings + re‑ranking. Includes API, tests, and a report generator for recruiter‑friendly results.

## Why it matters
- DS skills: data ingestion, feature/text processing, evaluation, reporting.
- AI engineering skills: serving with FastAPI, artifacts, reproducible builds, tests, and room for vector DB + LTR.

## Features
- TF‑IDF retrieval baseline (scikit‑learn) with cosine similarity.
- FastAPI service: POST /search by free text or stored resume.
- Pipeline scripts: build index and generate resume→job match reports (CSV/MD).
- Tests (pytest) for build, query, and reporting.

## Architecture
```
CSV jobs + text resumes --> Build (TF-IDF) --> artifacts/ {vectorizer, matrix, meta}
                |
                v
              FastAPI /search
                |
                v
          JSON results (top_k)
```

## Quickstart
1) Setup (Windows bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Build the index
```bash
python -m src.pipeline.build_index
```
3) Run API
```bash
uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000
```
4) Try it
- Resume on file:
```bash
curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d '{"resume_id":"resume_4","top_k":3}'
```
- Free text:
```bash
curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d '{"text":"machine learning engineer python aws","top_k":3}'
```

## Reporting
Generate top‑k matches for each resume:
```bash
python -m src.pipeline.report_matches
```
Outputs:
- artifacts/resume_job_matches.csv
- artifacts/resume_job_matches.md

## Project layout
- `config/config.yaml` — paths and settings
- `data/sample/` — example jobs and resumes
- `src/` — library code and API
- `artifacts/` — generated model/index files and reports
- `tests/` — pytest suite

## Resume‑ready bullets
- Built a job recommendation system (TF‑IDF baseline) with FastAPI serving; added pipelines for index build and reporting; 3 tests validate end‑to‑end flow.
- Structured for upgrades: sentence embeddings (vector DB), LTR re‑ranker (LightGBM/XGBoost), and evaluation (NDCG/MRR vs. keyword baseline).

## License
MIT — see `LICENSE`.
