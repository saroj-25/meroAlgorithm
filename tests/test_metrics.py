"""Retrieval, generation and fusion metric properties."""

import pytest

from src.evaluation.generation_metrics import grounding_rate, rouge_l, token_f1
from src.evaluation.retrieval_metrics import (evaluate_query, precision_at_k,
                                              reciprocal_rank, recall_at_k)
from src.retrieval.hybrid import reciprocal_rank_fusion


def test_precision_and_recall():
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = ["a", "b", "c"]
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "a"], ["a", "b"]) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_perfect_retrieval_scores_one():
    metrics = evaluate_query(["a", "b", "c"], ["a", "b", "c"], ks=(3,))
    assert metrics["P@3"] == 1.0 and metrics["MRR"] == 1.0 and metrics["nDCG@3"] == 1.0


def test_rrf_prefers_documents_found_by_both_retrievers():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 5]], k=60)
    # doc 2 is ranked second by both retrievers; docs 1 and 5 appear in one list only
    assert fused[2] > fused[1] and fused[2] > fused[5]


def test_rrf_weights_shift_the_ordering():
    unweighted = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60)
    weighted = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, weights=[1.0, 0.1])
    assert unweighted[1] == pytest.approx(unweighted[2])
    assert weighted[1] > weighted[2]


def test_token_f1_and_rouge_bounds():
    assert token_f1("merge sort is stable", "merge sort is stable") == pytest.approx(1.0)
    assert token_f1("merge sort", "quick heap") == 0.0
    assert 0.0 <= rouge_l("binary search halves the range", "binary search") <= 1.0


def test_grounding_rate_detects_ungrounded_text():
    chunks = [{"text": "merge sort is a stable divide and conquer sorting algorithm"}]
    assert grounding_rate("merge sort is stable", chunks) == pytest.approx(1.0)
    assert grounding_rate("quantum teleportation entangles qubits", chunks) < 0.3
