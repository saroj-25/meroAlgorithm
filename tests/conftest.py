"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture(scope="session")
def kb_path(cfg):
    path = cfg.path("paths.processed_dir") / "knowledge_base.jsonl"
    if not path.exists():
        pytest.skip("knowledge base not built; run `make data`")
    return path


@pytest.fixture(scope="session")
def retriever(cfg):
    from src.retrieval.hybrid import HybridRetriever

    try:
        return HybridRetriever.from_checkpoint(cfg)
    except FileNotFoundError:
        pytest.skip("index not built; run `make index`")
