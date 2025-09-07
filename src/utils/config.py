from __future__ import annotations
import os
import yaml
from dataclasses import dataclass


@dataclass
class Paths:
    jobs_csv: str
    resumes_dir: str
    artifacts_dir: str


@dataclass
class VectorizerCfg:
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.9


@dataclass
class IndexCfg:
    top_k_default: int = 5


@dataclass
class AppConfig:
    paths: Paths
    vectorizer: VectorizerCfg
    index: IndexCfg


def load_config(path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    paths = Paths(**raw["paths"])
    vec = VectorizerCfg(**raw.get("vectorizer", {}))
    # Ensure tuple type for sklearn params
    if isinstance(vec.ngram_range, list):
        vec.ngram_range = tuple(vec.ngram_range)  # type: ignore[assignment]
    idx = IndexCfg(**raw.get("index", {}))
    return AppConfig(paths=paths, vectorizer=vec, index=idx)
