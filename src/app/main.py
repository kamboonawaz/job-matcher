from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.index.vectorstore import TfidfJobIndex
from src.utils.config import load_config


app = FastAPI(title="Job Matcher API", version="0.1.0")


class SearchRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Free text query")
    resume_id: Optional[str] = Field(default=None, description="Filename stem under resumes_dir, e.g., resume_1")
    top_k: int = Field(default=5, ge=1, le=50)


@app.on_event("startup")
def load_index():
    global index, cfg
    cfg = load_config()
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.artifacts_dir)
    try:
        index = TfidfJobIndex.load(artifacts_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load index from {artifacts_dir}. Did you run the build step? Error: {e}")


@app.post("/search")
def search(req: SearchRequest):
    global index, cfg
    if req.text:
        query_text = req.text
    elif req.resume_id:
        resumes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cfg.paths.resumes_dir)
        path = os.path.join(resumes_dir, f"{req.resume_id}.txt")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Resume not found: {req.resume_id}")
        with open(path, "r", encoding="utf-8") as f:
            query_text = f.read()
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'resume_id'")

    try:
        results = index.query(query_text, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
