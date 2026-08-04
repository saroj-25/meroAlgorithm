"""Loss functions used by the trainable components.

1. ``softmax_cross_entropy`` - the objective of the character n-gram
   code-switch classifier (Section 3.3).  Implemented in numpy with its
   analytic gradient so the training loop in ``src/training`` is fully
   transparent and has no deep-learning dependency.

2. ``info_nce`` (a.k.a. MultipleNegativesRankingLoss) - the objective you would
   use to fine-tune the multilingual encoder on (Romanized Nepali query,
   English chunk) pairs, which is listed as future work in the paper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                          class_weights: np.ndarray | None = None
                          ) -> Tuple[float, np.ndarray]:
    """Mean cross-entropy and dL/dlogits.

    Parameters
    ----------
    logits : (n_samples, n_classes)
    targets : (n_samples,) integer class indices
    class_weights : optional (n_classes,) weights for imbalanced label sets
    """
    n = logits.shape[0]
    probs = softmax(logits)
    correct = probs[np.arange(n), targets]
    per_sample = -np.log(np.clip(correct, 1e-12, None))

    weights = np.ones(n) if class_weights is None else class_weights[targets]
    loss = float(np.sum(weights * per_sample) / max(np.sum(weights), 1e-12))

    grad = probs.copy()
    grad[np.arange(n), targets] -= 1.0
    grad *= weights[:, None]
    grad /= max(np.sum(weights), 1e-12)
    return loss, grad


def l2_penalty(weights: np.ndarray, strength: float) -> Tuple[float, np.ndarray]:
    """Ridge regularisation term and its gradient."""
    return 0.5 * strength * float(np.sum(weights ** 2)), strength * weights


def info_nce(query_emb: np.ndarray, positive_emb: np.ndarray,
             temperature: float = 0.05) -> Tuple[float, np.ndarray]:
    """In-batch contrastive loss for (query, positive-chunk) pairs.

    Every other positive in the batch acts as a negative, which is why this loss
    is so effective for retrieval fine-tuning: the batch supplies hard negatives
    for free.  Returns the loss and dL/d(similarity logits).
    """
    sims = (query_emb @ positive_emb.T) / temperature
    targets = np.arange(sims.shape[0])
    return softmax_cross_entropy(sims, targets)
