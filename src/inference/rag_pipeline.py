"""The online path of Figure 2: query -> answer.

    preprocess (normalize, detect register, optional transliteration expansion)
      -> hybrid retrieval (dense + BM25, RRF)
      -> cross-encoder re-ranking (top-20 -> top-5)
      -> pedagogical prompt construction
      -> generation (register-preserving, grounded, cited)
      -> post-processing (citation linking, latency + log record)

Everything is wired through one class so the CLI, the Streamlit app and the
evaluation scripts all exercise exactly the same code path.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..models.generator import BaseGenerator, build_generator
from ..preprocessing.language_id import load_lid
from ..preprocessing.query_pipeline import ProcessedQuery, preprocess_query
from ..retrieval.hybrid import HybridRetriever
from ..utils.config import Config
from ..utils.logging_utils import get_logger

log = get_logger("inference.rag")

_CITATION_RE = re.compile(r"\[([A-Z]{3}-(?:EN|CM)-\d{4})\]")


@dataclass
class RAGResponse:
    """Everything the UI, the logs and the evaluator need from one turn."""

    query: str
    answer: str
    chunks: List[dict]
    query_type: str
    language_profile: Dict[str, float]
    expanded: bool
    latency_s: float
    generator: str
    citations: List[str] = field(default_factory=list)
    grounded: bool = True

    def as_log_record(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "language_profile": {k: round(v, 3) for k, v in self.language_profile.items()},
            "retrieved_chunk_ids": [c["chunk_id"] for c in self.chunks],
            "citations": self.citations,
            "latency_s": round(self.latency_s, 3),
            "generator": self.generator,
            "grounded": self.grounded,
        }


class AlgoSathiRAG:
    """The deployed system: retriever + detector + generator behind one method."""

    def __init__(self, cfg: Config, retriever: HybridRetriever | None = None,
                 generator: BaseGenerator | None = None, detector=None):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever.from_checkpoint(cfg)
        self.detector = detector or load_lid(
            cfg.path("paths.checkpoints_dir") / "lid_model.pkl")
        self.generator = generator or build_generator(cfg)

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, top_k: int | None = None,
            force_language: str | None = None, use_dense: bool = True,
            use_bm25: bool = True, rerank: bool = True) -> RAGResponse:
        t0 = time.perf_counter()

        processed: ProcessedQuery = preprocess_query(
            question, self.detector, self.cfg, force_language=force_language)

        chunks = self.retriever.retrieve(processed, top_k=top_k, use_dense=use_dense,
                                         use_bm25=use_bm25, rerank=rerank)
        answer = self.generator.generate(processed.normalized, chunks,
                                         processed.language_profile)

        citations = _CITATION_RE.findall(answer)
        retrieved_ids = {c["chunk_id"] for c in chunks}
        # A citation that is not in the retrieved set is a hallucinated source.
        grounded = bool(citations) and all(c in retrieved_ids for c in citations)

        return RAGResponse(
            query=question, answer=answer, chunks=chunks,
            query_type=processed.query_type,
            language_profile=processed.language_profile,
            expanded=processed.expanded,
            latency_s=time.perf_counter() - t0,
            generator=self.generator.name,
            citations=citations, grounded=grounded)

    # ------------------------------------------------------------- utilities
    def retrieve_only(self, question: str, top_k: int | None = None) -> List[dict]:
        processed = preprocess_query(question, self.detector, self.cfg)
        return self.retriever.retrieve(processed, top_k=top_k)

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "AlgoSathiRAG":
        from ..utils.config import load_config

        return cls(load_config(config_path))
