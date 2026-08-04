"""BM25 correctness properties."""

import numpy as np

from src.models.bm25 import BM25Okapi, bm25_tokenize

CORPUS = [
    "the kruskal algorithm builds a minimum spanning tree by sorting edges",
    "the dijkstra algorithm finds shortest paths using a priority queue",
    "the avl tree algorithm rebalances with rotations after insertion",
    "the sorting algorithm compares elements pairwise",
]


def test_tokenizer_keeps_alphanumerics():
    assert bm25_tokenize("O(n log n) and dp[i][j]!") == ["o", "n", "log", "n", "and", "dp", "i", "j"]


def test_exact_term_ranks_first():
    bm25 = BM25Okapi(CORPUS)
    assert bm25.top_k("avl rotations", k=1)[0][0] == 2


def test_rare_term_scores_higher_than_common_term():
    bm25 = BM25Okapi(CORPUS)
    rare = bm25.get_scores("kruskal").max()
    common = bm25.get_scores("algorithm").max()   # appears in every document
    assert rare > common, "IDF must down-weight terms appearing in many documents"


def test_scores_are_non_negative_and_sized_correctly():
    bm25 = BM25Okapi(CORPUS)
    scores = bm25.get_scores("spanning tree")
    assert scores.shape == (len(CORPUS),)
    assert np.all(scores >= 0)


def test_unknown_term_gives_zero_scores():
    bm25 = BM25Okapi(CORPUS)
    assert bm25.get_scores("zzzznotaword").sum() == 0.0
