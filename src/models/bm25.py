"""BM25Okapi lexical retriever (self-contained, no external dependency).

Why implement it here?  Because the paper's hybrid retriever depends on BM25
scoring *rare transliterated technical terms* (AVL, Kruskal, Floyd-Warshall)
that a dense encoder often misses, and it is worth seeing the formula rather
than importing it:

    score(q, d) = sum over terms t in q of
                  IDF(t) * f(t,d) * (k1 + 1)
                  ------------------------------------------
                  f(t,d) + k1 * (1 - b + b * |d| / avgdl)

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )

with k1 = 1.5 and b = 0.75 (Table 2 of the paper).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097F]+")


def bm25_tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer that keeps Devanagari and alphanumerics."""
    return _TOKEN_RE.findall(text.lower())


class BM25Okapi:
    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.docs: List[List[str]] = [bm25_tokenize(doc) for doc in corpus]
        self.n_docs = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # term -> {doc_index: term frequency}
        self.postings: Dict[str, Dict[int, int]] = {}
        for idx, doc in enumerate(self.docs):
            for term, freq in Counter(doc).items():
                self.postings.setdefault(term, {})[idx] = freq

        # Pre-compute IDF once; it does not depend on the query.
        self.idf: Dict[str, float] = {
            term: math.log((self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
            for term, posting in self.postings.items()
        }

    # ------------------------------------------------------------------ score
    def get_scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for one query (dense numpy vector)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in bm25_tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_idx, freq in posting.items():
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / (freq + norm[doc_idx])
        return scores

    def top_k(self, query: str, k: int = 50):
        """Return ``[(doc_index, score), ...]`` sorted by decreasing score."""
        scores = self.get_scores(query)
        if k >= self.n_docs:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
