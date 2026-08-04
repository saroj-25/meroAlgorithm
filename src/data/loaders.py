"""Typed loaders shared by scripts, tests and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..utils.config import Config
from ..utils.io import read_jsonl


def load_knowledge_base(cfg: Config) -> List[dict]:
    return read_jsonl(cfg.path("paths.processed_dir") / "knowledge_base.jsonl")


def load_knowledge_base_df(cfg: Config) -> pd.DataFrame:
    return pd.DataFrame(load_knowledge_base(cfg))


def load_eval_queries(cfg: Config, query_type: str | None = None) -> List[dict]:
    rows = read_jsonl(cfg.path("paths.processed_dir") / "eval_queries.jsonl")
    return [r for r in rows if query_type in (None, r["query_type"])]


def load_lid_dataset(cfg: Config) -> List[dict]:
    return read_jsonl(cfg.path("paths.processed_dir") / "lid_dataset.jsonl")


def load_study_scores(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else cfg.path("paths.processed_dir") / "study_scores.csv"
    df = pd.read_csv(path)
    required = {"student_id", "group", "pre_test", "post_test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["gain"] = df["post_test"] - df["pre_test"]
    return df


def load_interaction_logs(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.path("paths.processed_dir") / "interaction_logs.csv")


def load_satisfaction(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.path("paths.processed_dir") / "satisfaction_responses.csv")
