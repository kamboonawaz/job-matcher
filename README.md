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

          # 🧑‍💼 Job Matcher — AI-Powered Resume-to-Job Recommender

          [![CI](https://github.com/kamboonawaz/job-matcher-private/actions/workflows/ci.yml/badge.svg)](https://github.com/kamboonawaz/job-matcher-private/actions)
          [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

          > **Showcase your Data Science & AI Engineering skills:**
          > - End-to-end ML pipeline: data ingestion, feature engineering, model building, API, reporting, and CI/CD.
          > - Modern stack: Python, FastAPI, scikit-learn, pytest, Docker, GitHub Actions.
          > - Ready for extension: swap in embeddings, add re-ranking, or deploy as a microservice.

          ---

          ## 🚀 Project Highlights

          - **AI-powered job recommender:** Ranks job postings for a candidate using TF‑IDF, cosine similarity, and a FastAPI backend.
          - **Production-ready structure:** Modular code, config-driven, with tests, reporting, and Docker support.
          - **Recruiter-facing:** Clear, reproducible results; easy to extend with embeddings, LTR, or cloud deployment.

          ---

          ## 🏆 Skills Demonstrated

          - Data wrangling (Pandas, CSV, text parsing)
          - Feature engineering (TF‑IDF, n-grams)
          - Model serving (FastAPI, Docker)
          - Automated testing (pytest, smoke tests)
          - Reporting (CSV, Markdown, batch scoring)
          - CI/CD (GitHub Actions)
          - Clean code, config management, reproducibility

          ---

          ## 🛠️ Quickstart

          ### 1. Setup (Windows bash)
          ```bash
          python -m venv .venv
          source .venv/Scripts/activate
          pip install --upgrade pip
          pip install -r requirements.txt
          ```

          ### 2. Build the index
          ```bash
          python -m src.pipeline.build_index
          ```

          ### 3. Run the API
          ```bash
          uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000
          ```

          ### 4. Try it out
          - Search with a stored resume:
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

          ### 5. Batch reporting
          ```bash
          python -m src.pipeline.report_matches
          # See artifacts/resume_job_matches.md for top matches per resume
          ```

          ---

          ## 📦 Project Structure

          - `config/config.yaml` — paths and settings
          - `data/sample/` — example jobs and resumes
          - `src/` — library code and API
          - `artifacts/` — auto-generated model/index files
          - `tests/` — smoke and batch tests

          ---

          ## 🧩 Extending for AI/DS Roles

          - Swap TF‑IDF for sentence embeddings (see `src/index/vectorstore.py`)
          - Add a learning-to-rank model (LightGBM/XGBoost) with features like skills overlap
          - Deploy with Docker or to the cloud (Azure, AWS, GCP)
          - Integrate with a web UI or Slack bot

          ---

          ## 📄 License

          MIT — see [LICENSE](LICENSE)
