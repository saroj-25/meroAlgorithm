"""Token-level code-switch detection (Section 3.3).

The paper trains "a character-level n-gram classifier ... on a held-out subset of
4,000 Romanized Nepali, English, and code-mixed sentences" that emits, for each
token, a probability of being Romanized Nepali, English, or script-ambiguous
(technical identifiers such as ``O(n log n)``, ``DFS`` or ``i++``), reaching a
token-level F1 of 0.91.

This module implements exactly that, as a **multinomial logistic regression over
character n-gram TF-IDF features**, trained with the numpy loop in
``src/training/train_lid.py``.  Character features are the whole point: Romanized
Nepali has no standard orthography, so a model keyed on whole words would fail on
every unseen spelling variant, while ``-ncha``/``-nchha``/``-nxa`` endings are
learnable at the character level.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..utils.logging_utils import get_logger
from .lexicons import LABELS, clean_token, is_technical, rule_language_profile, tokenize

log = get_logger("preprocessing.lid")


class CharNGramLID:
    """Character n-gram softmax classifier for token-level language ID."""

    def __init__(self, ngram_range=(1, 4), max_features: int = 30000,
                 labels: Sequence[str] = tuple(LABELS)):
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.labels = list(labels)
        self.vectorizer = None
        self.W: np.ndarray | None = None      # (n_features, n_classes)
        self.b: np.ndarray | None = None      # (n_classes,)
        self.history: List[dict] = []          # filled by the trainer

    # ------------------------------------------------------------- features
    def fit_vectorizer(self, tokens: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range,
            max_features=self.max_features, lowercase=True, sublinear_tf=True,
            min_df=1)
        matrix = self.vectorizer.fit_transform([f" {t} " for t in tokens])
        log.info("LID vectorizer: %d tokens -> %d character n-gram features",
                 matrix.shape[0], matrix.shape[1])
        return matrix

    def transform(self, tokens: Sequence[str]):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted")
        return self.vectorizer.transform([f" {t} " for t in tokens])

    # -------------------------------------------------------------- predict
    def _logits(self, tokens: Sequence[str]) -> np.ndarray:
        X = self.transform(tokens)
        return np.asarray(X @ self.W) + self.b

    def predict_proba(self, tokens: Sequence[str]) -> np.ndarray:
        from ..models.losses import softmax

        if self.W is None:
            raise RuntimeError("Model not trained")
        return softmax(self._logits(list(tokens)))

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Predict a label per token, with a hard rule for technical tokens.

        Technical identifiers are handled by the deterministic rule rather than
        the classifier: ``O(n log n)`` is not a language, and treating it as one
        would corrupt the register estimate for otherwise-Nepali queries.
        """
        tokens = list(tokens)
        if not tokens:
            return []
        probs = self.predict_proba(tokens)
        out = []
        for token, row in zip(tokens, probs):
            out.append("ambiguous" if is_technical(token)
                       else self.labels[int(np.argmax(row))])
        return out

    def language_profile(self, text: str) -> Dict[str, float]:
        """Proportion of nepali / english / ambiguous tokens in a query."""
        tokens = tokenize(text)
        if not tokens:
            return {lab: (1.0 if lab == "ambiguous" else 0.0) for lab in self.labels}
        if self.W is None:                       # untrained -> rule fallback
            return rule_language_profile(text)
        labels = self.predict_tokens(tokens)
        return {lab: labels.count(lab) / len(labels) for lab in self.labels}

    def classify_query(self, text: str, nepali_threshold: float = 0.15,
                       mostly_threshold: float = 0.55) -> str:
        """Map a profile onto the paper's three query types.

        ``romanized_nepali`` (mostly Nepali) | ``code_mixed`` | ``english``
        """
        profile = self.language_profile(text)
        nepali, english = profile.get("nepali", 0.0), profile.get("english", 0.0)
        if nepali >= mostly_threshold and english < 0.25:
            return "romanized_nepali"
        if nepali >= nepali_threshold:
            return "code_mixed"
        return "english"

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("Saved LID checkpoint -> %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "CharNGramLID":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class RuleBasedLID:
    """Zero-training fallback with the same interface (lexicon lookups only)."""

    labels = list(LABELS)

    def language_profile(self, text: str) -> Dict[str, float]:
        return rule_language_profile(text)

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        from .lexicons import rule_label

        return [rule_label(t) for t in tokens]

    def classify_query(self, text: str, nepali_threshold: float = 0.15,
                       mostly_threshold: float = 0.55) -> str:
        profile = self.language_profile(text)
        nepali, english = profile["nepali"], profile["english"]
        if nepali >= mostly_threshold and english < 0.25:
            return "romanized_nepali"
        if nepali >= nepali_threshold:
            return "code_mixed"
        return "english"


def load_lid(checkpoint: str | Path | None):
    """Load a trained checkpoint, or return the rule-based fallback."""
    if checkpoint and Path(checkpoint).exists():
        try:
            return CharNGramLID.load(checkpoint)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not load LID checkpoint (%s); using rules", exc)
    log.info("No LID checkpoint found; using the rule-based detector")
    return RuleBasedLID()
