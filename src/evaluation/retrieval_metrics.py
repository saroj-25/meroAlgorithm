"""Retrieval metrics: P@k, MRR, R@k, nDCG@k .
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / float(len(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document (0 if none was retrieved)."""
    relevant = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_query(retrieved: Sequence[str], relevant: Iterable[str],
                   ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    relevant = list(relevant)
    out: Dict[str, float] = {"MRR": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"R@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"nDCG@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return out


def aggregate(per_query: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric over queries (macro-average, as in the paper)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: float(sum(row[key] for row in per_query) / len(per_query))
            for key in keys}
