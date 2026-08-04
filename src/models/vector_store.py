"""Vector store with a FAISS-HNSW backend and a numpy brute-force fallback.

Paper configuration (Table 2): FAISS HNSW with M = 32, efConstruction = 200,
efSearch = 64, cosine similarity over 768-d vectors.  For 1,252 chunks a flat
search is already sub-millisecond, so the numpy fallback is exact and fast; the
HNSW path exists because it is what scales to the larger corpora the paper
anticipates.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..utils.logging_utils import get_logger

log = get_logger("models.vector_store")


class BaseVectorStore:
    backend = "base"

    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseVectorStore":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class NumpyFlatIndex(BaseVectorStore):
    """Exact cosine search: a single matrix-vector product."""

    backend = "numpy_flat"

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = np.vstack([self.vectors, vectors / norms])

    def search(self, query_vector: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        if self.vectors.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(q) or 1.0
        sims = self.vectors @ (q / norm)
        k = min(k, sims.shape[0])
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in order]


class FaissHNSWIndex(BaseVectorStore):
    """Paper-faithful FAISS HNSW index (inner product on normalised vectors)."""

    backend = "faiss_hnsw"

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200,
                 ef_search: int = 64):
        import faiss  # heavy optional import

        self.dim, self.m = dim, m
        self.ef_construction, self.ef_search = ef_construction, ef_search
        self._index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray) -> None:
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        import faiss

        q = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # faiss indexes need their own serialiser
    def __getstate__(self):
        import faiss

        state = {k: v for k, v in self.__dict__.items() if k != "_index"}
        state["_serialized"] = faiss.serialize_index(self._index)
        return state

    def __setstate__(self, state):
        import faiss

        blob = state.pop("_serialized")
        self.__dict__.update(state)
        self._index = faiss.deserialize_index(blob)


def build_vector_store(cfg, dim: int) -> BaseVectorStore:
    backend = cfg.get("vector_store.backend", "auto")
    offline = cfg.get("project.mode", "auto") == "offline"
    if backend in ("auto", "faiss_hnsw") and not offline:
        try:
            store = FaissHNSWIndex(dim,
                                   m=cfg.get("vector_store.hnsw_m", 32),
                                   ef_construction=cfg.get("vector_store.ef_construction", 200),
                                   ef_search=cfg.get("vector_store.ef_search", 64))
            log.info("Using FAISS HNSW index (M=%d, efSearch=%d)",
                     store.m, store.ef_search)
            return store
        except Exception as exc:
            if backend == "faiss_hnsw":
                raise
            log.warning("faiss unavailable (%s); using exact numpy index", exc)
    return NumpyFlatIndex(dim)
