from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load


@dataclass
class CorpusItem:
    job_id: str
    title: str
    company: str
    location: str
    description: str


class TfidfJobIndex:
    def __init__(self, vectorizer: TfidfVectorizer, X, meta: List[CorpusItem]):
        self.vectorizer = vectorizer
        self.X = X
        self.meta = meta

    @staticmethod
    def preprocess(text: str) -> str:
        return (text or "").lower()

    @classmethod
    def from_corpus(cls, docs: List[str], meta: List[CorpusItem], *, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )
        X = vectorizer.fit_transform([cls.preprocess(d) for d in docs])
        return cls(vectorizer, X, meta)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self.vectorizer.transform([self.preprocess(text)])
        sims = cosine_similarity(q, self.X)[0]
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            m = self.meta[int(i)]
            results.append({
                "job_id": m.job_id,
                "title": m.title,
                "company": m.company,
                "location": m.location,
                "score": float(sims[i]),
            })
        return results

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        dump(self.vectorizer, os.path.join(dir_path, "vectorizer.joblib"))
        dump(self.X, os.path.join(dir_path, "tfidf_matrix.joblib"))
        # serialize meta
        meta_dicts = [m.__dict__ for m in self.meta]
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_dicts, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "TfidfJobIndex":
        vectorizer = load(os.path.join(dir_path, "vectorizer.joblib"))
        X = load(os.path.join(dir_path, "tfidf_matrix.joblib"))
        with open(os.path.join(dir_path, "meta.json"), "r", encoding="utf-8") as f:
            meta_dicts = json.load(f)
        meta = [CorpusItem(**d) for d in meta_dicts]
        return cls(vectorizer, X, meta)
