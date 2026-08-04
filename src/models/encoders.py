

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from ..utils.logging_utils import get_logger

log = get_logger("models.encoder")


# --------------------------------------------------------------------------- #
class BaseEncoder:
    """Common interface: ``fit`` (optional), ``encode``, ``save``, ``load``."""

    name = "base"
    dim = 0

    def fit(self, corpus: Sequence[str]) -> "BaseEncoder":
        return self

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseEncoder":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------- #
class TfidfSvdEncoder(BaseEncoder):
    """Offline encoder: char n-gram + word TF-IDF, reduced with truncated SVD.

    This is *latent semantic indexing*.  It has no cross-lingual pre-training,
    so its Romanized-Nepali-to-English transfer comes only from shared technical
    tokens - which is precisely why the hybrid (lexical + dense) retriever and
    the transliteration expansion matter so much in the offline configuration.
    """

    name = "tfidf_svd"

    def __init__(self, dim: int = 768, char_ngrams=(3, 5), word_ngrams=(1, 2),
                 min_df: int = 1, random_state: int = 42):
        self.dim = dim
        self.char_ngrams = tuple(char_ngrams)
        self.word_ngrams = tuple(word_ngrams)
        self.min_df = min_df
        self.random_state = random_state
        self._vectorizer = None
        self._svd = None

    def fit(self, corpus: Sequence[str]) -> "TfidfSvdEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vectorizer = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams,
                                     min_df=self.min_df, sublinear_tf=True,
                                     lowercase=True, max_features=200_000)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=self.word_ngrams,
                                     min_df=self.min_df, sublinear_tf=True,
                                     lowercase=True, max_features=100_000)),
        ])
        matrix = self._vectorizer.fit_transform(list(corpus))
        n_comp = int(min(self.dim, matrix.shape[1] - 1, max(2, matrix.shape[0] - 1)))
        if n_comp != self.dim:
            log.warning("Reducing SVD dimension from %d to %d (corpus too small)",
                        self.dim, n_comp)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self._svd.fit(matrix)
        self.dim = n_comp
        log.info("Fitted TfidfSvdEncoder: %d docs -> %d features -> %d dims "
                 "(explained variance %.3f)", matrix.shape[0], matrix.shape[1],
                 n_comp, float(self._svd.explained_variance_ratio_.sum()))
        return self

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfSvdEncoder must be fitted (or loaded) before use")
        texts = list(texts)
        vectors = self._svd.transform(self._vectorizer.transform(texts))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
class SentenceTransformerEncoder(BaseEncoder):
    """Paper-faithful backend (``paraphrase-multilingual-mpnet-base-v2``)."""

    name = "sentence_transformers"

    def __init__(self, model_name: str, max_seq_length: int = 128,
                 normalize: bool = True):
        from sentence_transformers import SentenceTransformer  # heavy import

        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = max_seq_length
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        vectors = self._model.encode(list(texts), batch_size=batch_size,
                                     convert_to_numpy=True,
                                     normalize_embeddings=self.normalize,
                                     show_progress_bar=False)
        return vectors.astype(np.float32)

    # The torch model is not picklable across environments: save only the name.
    def __getstate__(self):
        return {"model_name": self.model_name, "normalize": self.normalize,
                "dim": self.dim}

    def __setstate__(self, state):
        from sentence_transformers import SentenceTransformer

        self.__dict__.update(state)
        self._model = SentenceTransformer(state["model_name"])


# --------------------------------------------------------------------------- #
def build_encoder(cfg, corpus: Sequence[str] | None = None) -> BaseEncoder:
    """Instantiate the encoder requested by the config, honouring offline mode."""
    backend = cfg.get("encoder.backend", "auto")
    offline = cfg.get("project.mode", "auto") == "offline"

    if backend in ("auto", "sentence_transformers") and not offline:
        try:
            enc = SentenceTransformerEncoder(
                cfg.get("encoder.model_name"),
                max_seq_length=cfg.get("encoder.max_seq_length", 128),
                normalize=cfg.get("encoder.normalize", True),
            )
            log.info("Using sentence-transformers encoder: %s", enc.model_name)
            return enc
        except Exception as exc:
            if backend == "sentence_transformers":
                raise
            log.warning("sentence-transformers unavailable (%s); "
                        "falling back to the offline TF-IDF+SVD encoder", exc)

    enc = TfidfSvdEncoder(dim=cfg.get("encoder.dim", 768),
                          random_state=cfg.get("project.seed", 42))
    if corpus is not None:
        enc.fit(corpus)
    return enc
