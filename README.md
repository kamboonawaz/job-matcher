# Job Matcher (Private)

Lightweight job recommendation service that ranks job postings for a candidate using text relevance (TF‑IDF + cosine). Designed as a foundation to extend with embeddings (sentence-transformers), re-ranking (LightGBM), and a vector DB.

## Features
- Data ingestion from CSV (jobs) and plain text resumes.
- TF‑IDF vectorizer + cosine similarity retrieval.
- FastAPI endpoint to search by free text or stored resume.
- Reproducible build script and small sample dataset.

## Project layout
- `config/config.yaml` — paths and settings
- `data/sample/` — example jobs and resumes
- `src/` — library code and API
- `artifacts/` — auto-generated model/index files
- `tests/` — minimal smoke test

## Quickstart

### 1) Setup (Windows bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Build the index
```bash
python -m src.pipeline.build_index
```
This will create `artifacts/` with the trained TF‑IDF model, nearest-neighbor index, and metadata.

### 3) Run API
```bash
uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4) Try it
- Search with resume on file (created from `data/sample/resumes/resume_1.txt`):
```bash
curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"resume_id":"resume_1", "top_k":3}' | jq
```
- Or search with free text:
```bash
curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"text":"machine learning engineer python aws", "top_k":3}' | jq
```

## Extending to embeddings + re-ranking
- Swap TF‑IDF for sentence embeddings (e.g., `sentence-transformers`) in `src/index/vectorstore.py`.
- Add a learning-to-rank model in `src/models/reranker.py` using features like BM25/TF‑IDF score, seniority match, skills overlap; train with XGBoost/LightGBM.

## Notes
- This repo is local/private by default. If you create a Git repo, ensure it’s private and keep sample data non-sensitive.
