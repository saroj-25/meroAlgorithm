

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

from ..models.bm25 import BM25Okapi
from ..models.encoders import BaseEncoder
from ..models.reranker import BaseReranker, build_reranker
from ..models.vector_store import BaseVectorStore
from ..preprocessing.query_pipeline import ProcessedQuery
from ..utils.config import Config
from ..utils.io import read_jsonl
from ..utils.logging_utils import get_logger

log = get_logger("retrieval.hybrid")


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[int]], k: int = 60,
                           weights: Sequence[float] | None = None) -> Dict[int, float]:
    """Fuse ranked document-index lists into a single score per document.

    ``weights`` generalises RRF to the weighted form

        RRF(d) = sum over retrievers r of  w_r / (k + rank_r(d))

    which matters when the two retrievers are of very unequal strength (see
    ``docs/findings.md``: with a weak offline encoder, unweighted fusion can be
    worse than the better retriever alone).  Default weights are all 1.0, which
    is the standard formulation used in the paper.
    """
    weights = list(weights or [1.0] * len(ranked_lists))
    fused: Dict[int, float] = {}
    for ranking, weight in zip(ranked_lists, weights):
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return fused


class HybridRetriever:
    """Loads the offline artefacts and serves the online retrieval path."""

    def __init__(self, cfg: Config, encoder: BaseEncoder, store: BaseVectorStore,
                 bm25: BM25Okapi, chunks: List[dict], reranker: BaseReranker | None = None):
        self.cfg = cfg
        self.encoder = encoder
        self.store = store
        self.bm25 = bm25
        self.chunks = chunks
        self.reranker = reranker or build_reranker(cfg, idf=bm25.idf)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_checkpoint(cls, cfg: Config, path: str | Path | None = None) -> "HybridRetriever":
        path = Path(path or (cfg.path("paths.checkpoints_dir") / "index"))
        if not (path / "encoder.pkl").exists():
            raise FileNotFoundError(
                f"No index at {path}. Run: python -m src.retrieval.index_builder")
        encoder = BaseEncoder.load(path / "encoder.pkl")
        store = BaseVectorStore.load(path / "vector_store.pkl")
        with open(path / "bm25.pkl", "rb") as fh:
            bm25 = pickle.load(fh)
        chunks = read_jsonl(path / "chunks.jsonl")
        log.info("Loaded index: %d chunks, encoder=%s, store=%s",
                 len(chunks), encoder.name, store.backend)
        return cls(cfg, encoder, store, bm25, chunks)

    # -------------------------------------------------------------- retrieve
    def retrieve(self, query: ProcessedQuery | str, top_k: int | None = None,
                 use_dense: bool = True, use_bm25: bool = True,
                 rerank: bool = True, dense_weight: float | None = None,
                 bm25_weight: float | None = None) -> List[dict]:
        """Return the final chunk list, each annotated with its scores.

        The ``use_dense`` / ``use_bm25`` / ``rerank`` switches exist so that the
        ablation study in Section 6.1 is one function call away.
        """
        cfg = self.cfg
        if isinstance(query, str):                       # convenience path
            from ..preprocessing.language_id import RuleBasedLID
            from ..preprocessing.query_pipeline import preprocess_query

            query = preprocess_query(query, RuleBasedLID(), cfg)

        final_k = top_k or cfg.get("retrieval.final_top_k", 5)
        dense_k = cfg.get("retrieval.dense_top_k", 50)
        lex_k = cfg.get("retrieval.bm25_top_k", 50)
        rerank_k = cfg.get("retrieval.rerank_top_k", 20)
        rrf_k = cfg.get("retrieval.rrf_k", 60)

        w_dense = dense_weight if dense_weight is not None else \
            cfg.get("retrieval.fusion_weights.dense", 1.0)
        w_bm25 = bm25_weight if bm25_weight is not None else \
            cfg.get("retrieval.fusion_weights.bm25", 1.0)

        rankings: List[List[int]] = []
        weights: List[float] = []
        dense_scores: Dict[int, float] = {}
        lexical_scores: Dict[int, float] = {}

        if use_dense:
            vector = self.encoder.encode([query.encode_text])[0]
            hits = self.store.search(vector, k=dense_k)
            dense_scores = {i: s for i, s in hits}
            rankings.append([i for i, _ in hits])
            weights.append(w_dense)

        if use_bm25:
            hits = self.bm25.top_k(query.lexical_text, k=lex_k)
            lexical_scores = {i: s for i, s in hits}
            rankings.append([i for i, _ in hits])
            weights.append(w_bm25)

        if not rankings:
            raise ValueError("At least one of use_dense / use_bm25 must be True")

        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
        order = sorted(fused, key=lambda i: -fused[i])

        # Register-aware boost: when the learner writes in Romanized Nepali or
        # code-mixes, the instructor-authored code-mixed chunks are usually the
        # better teaching material, so they get a small additive bonus.  This is
        # an addition to the paper's pipeline, ablated in experiments/.
        boost = cfg.get("retrieval.register_boost", 0.0)
        register_boosted = boost > 0 and query.query_type in ("romanized_nepali",
                                                              "code_mixed")

        candidates = []
        for idx in order[:max(rerank_k, final_k)]:
            chunk = dict(self.chunks[idx])
            chunk.update({
                "doc_index": idx,
                "dense_score": float(dense_scores.get(idx, 0.0)),
                "bm25_score": float(lexical_scores.get(idx, 0.0)),
                "fusion_score": float(fused[idx]),
            })
            if register_boosted and chunk.get("language") == "cm":
                chunk["fusion_score"] *= (1.0 + boost)
                chunk["register_boosted"] = True
            candidates.append(chunk)
        candidates.sort(key=lambda c: -c["fusion_score"])

        if rerank and self.reranker is not None:
            results = self.reranker.rerank(query.normalized, candidates, top_k=final_k)
        else:
            results = candidates[:final_k]
            for r in results:
                r.setdefault("rerank_score", r["fusion_score"])

        for rank, chunk in enumerate(results, start=1):
            chunk["rank"] = rank
        return results
