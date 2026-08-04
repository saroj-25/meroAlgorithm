from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from ..utils.logging_utils import get_logger
from .bm25 import bm25_tokenize

log = get_logger("models.reranker")


def _char_ngrams(text: str, n: int = 3) -> Counter:
    text = " " + text.lower().strip() + " "
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def _dice(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    return 2.0 * inter / (sum(a.values()) + sum(b.values()))


class BaseReranker:
    name = "base"

    def rerank(self, query: str, candidates: Sequence[dict], top_k: int = 5) -> List[dict]:
        raise NotImplementedError


class LexicalReranker(BaseReranker):
    """Offline re-ranker. ``idf`` maps a term to its inverse document frequency."""

    name = "lexical"

    def __init__(self, idf: Dict[str, float] | None = None, w_dense: float = 0.50,
                 w_overlap: float = 0.20, w_char: float = 0.15, w_prior: float = 0.15):
        self.idf = idf or {}
        self.w_dense = w_dense
        self.w_overlap, self.w_char, self.w_prior = w_overlap, w_char, w_prior

    def _pair_score(self, query: str, text: str) -> float:
        q_terms = set(bm25_tokenize(query))
        d_terms = Counter(bm25_tokenize(text))
        if not q_terms:
            return 0.0
        # IDF-weighted overlap coefficient: rare shared terms count for more.
        num = sum(self.idf.get(t, 1.0) * min(1, d_terms.get(t, 0)) for t in q_terms)
        den = sum(self.idf.get(t, 1.0) for t in q_terms) or 1.0
        overlap = num / den
        char_sim = _dice(_char_ngrams(query), _char_ngrams(text[:600]))
        return self.w_overlap * overlap + self.w_char * char_sim

    def rerank(self, query: str, candidates: Sequence[dict], top_k: int = 5) -> List[dict]:
        """Blend the semantic (dense) signal with lexical evidence.

        A purely lexical re-ranker would *undo* the cross-lingual work done by the
        dense retriever - it scored measurably worse than no re-ranking at all on
        Romanized Nepali queries in our ablation - so the dense score is kept as
        the dominant term and the lexical terms act as tie-breakers on rare
        technical strings.
        """
        if not candidates:
            return []
        prior_max = max(c.get("fusion_score", 0.0) for c in candidates) or 1.0
        dense_vals = [c.get("dense_score", 0.0) for c in candidates]
        d_lo, d_hi = min(dense_vals), max(dense_vals)
        d_range = (d_hi - d_lo) or 1.0

        scored = []
        for cand in candidates:
            prior = cand.get("fusion_score", 0.0) / prior_max
            dense_norm = (cand.get("dense_score", 0.0) - d_lo) / d_range
            score = (self.w_dense * dense_norm
                     + self._pair_score(query, cand["text"])
                     + self.w_prior * prior)
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda c: -c["rerank_score"])
        return scored[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Paper-faithful cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``)."""

    name = "cross_encoder"

    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: Sequence[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"][:2000]) for c in candidates]
        scores = self._model.predict(pairs)
        out = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            out.append(item)
        out.sort(key=lambda c: -c["rerank_score"])
        return out[:top_k]


def build_reranker(cfg, idf: Dict[str, float] | None = None) -> BaseReranker:
    backend = cfg.get("retrieval.reranker.backend", "auto")
    offline = cfg.get("project.mode", "auto") == "offline"
    if backend in ("auto", "cross_encoder") and not offline:
        try:
            rr = CrossEncoderReranker(cfg.get("retrieval.reranker.model_name"))
            log.info("Using cross-encoder re-ranker: %s", rr.model_name)
            return rr
        except Exception as exc:
            if backend == "cross_encoder":
                raise
            log.warning("cross-encoder unavailable (%s); using lexical re-ranker", exc)
    return LexicalReranker(idf=idf)
