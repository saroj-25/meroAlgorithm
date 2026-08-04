"""A transparent mini-batch trainer for the language-ID classifier.

Everything a training loop normally hides behind ``model.fit()`` is written out
here: forward pass, loss, analytic gradient, parameter update, validation pass,
early stopping and checkpointing.  It runs on sparse character n-gram features
with numpy only, so students can read it end to end without a GPU or a deep
learning framework.

Optimiser: mini-batch gradient descent with a simple 1/sqrt(t) learning-rate
decay and L2 (ridge) regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..models.losses import l2_penalty, softmax, softmax_cross_entropy
from ..utils.logging_utils import get_logger

log = get_logger("training.trainer")


@dataclass
class TrainingHistory:
    """Per-epoch curves, consumed by the plots and the Streamlit training page."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)

    def append(self, epoch: int, **metrics: float) -> None:
        self.epochs.append(epoch)
        for key, value in metrics.items():
            getattr(self, key).append(float(value))

    def as_records(self) -> List[Dict[str, float]]:
        return [
            {"epoch": e, "train_loss": tl, "val_loss": vl,
             "train_acc": ta, "val_acc": va, "val_macro_f1": f1}
            for e, tl, vl, ta, va, f1 in zip(
                self.epochs, self.train_loss, self.val_loss,
                self.train_acc, self.val_acc, self.val_macro_f1)
        ]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 - the metric reported in Section 3.3."""
    scores = []
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores))


class SoftmaxRegressionTrainer:
    """Multinomial logistic regression trained by mini-batch gradient descent."""

    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.5,
                 l2: float = 1e-4, batch_size: int = 128, epochs: int = 40,
                 patience: int = 6, class_weights: np.ndarray | None = None,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_features, n_classes), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)
        self.lr = learning_rate
        self.l2 = l2
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.history = TrainingHistory()

    # ------------------------------------------------------------- forward
    def logits(self, X) -> np.ndarray:
        return np.asarray(X @ self.W) + self.b

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.logits(X), axis=1)

    def evaluate(self, X, y: np.ndarray) -> Tuple[float, float, float]:
        """Return (loss, accuracy, macro-F1) on a dataset - the validation pass."""
        logits = self.logits(X)
        loss, _ = softmax_cross_entropy(logits, y, self.class_weights)
        reg, _ = l2_penalty(self.W, self.l2)
        preds = np.argmax(logits, axis=1)
        return loss + reg, float(np.mean(preds == y)), macro_f1(y, preds, self.n_classes)

    # -------------------------------------------------------------- fit
    def fit(self, X_train, y_train: np.ndarray, X_val, y_val: np.ndarray,
            verbose: bool = True) -> TrainingHistory:
        n = X_train.shape[0]
        best_val = np.inf
        best_params = (self.W.copy(), self.b.copy())
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(n)
            epoch_loss, seen = 0.0, 0

            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb = X_train[idx], y_train[idx]

                # ---- forward + loss ---------------------------------------
                loss, dlogits = softmax_cross_entropy(self.logits(Xb), yb,
                                                      self.class_weights)
                reg_loss, dreg = l2_penalty(self.W, self.l2)

                # ---- backward ---------------------------------------------
                grad_W = np.asarray(Xb.T @ dlogits) + dreg
                grad_b = dlogits.sum(axis=0)

                # ---- update (decayed step size) ---------------------------
                lr = self.lr / np.sqrt(epoch)
                self.W -= lr * grad_W
                self.b -= lr * grad_b

                epoch_loss += (loss + reg_loss) * len(idx)
                seen += len(idx)

            train_loss = epoch_loss / max(seen, 1)
            train_acc = float(np.mean(self.predict(X_train) == y_train))
            val_loss, val_acc, val_f1 = self.evaluate(X_val, y_val)
            self.history.append(epoch, train_loss=train_loss, val_loss=val_loss,
                                train_acc=train_acc, val_acc=val_acc,
                                val_macro_f1=val_f1)
            if verbose:
                log.info("epoch %2d | train_loss %.4f acc %.3f | val_loss %.4f "
                         "acc %.3f macroF1 %.3f",
                         epoch, train_loss, train_acc, val_loss, val_acc, val_f1)

            # ---- early stopping on validation loss -------------------------
            if val_loss < best_val - 1e-5:
                best_val, bad_epochs = val_loss, 0
                best_params = (self.W.copy(), self.b.copy())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    log.info("Early stopping at epoch %d (best val_loss %.4f)",
                             epoch, best_val)
                    break

        self.W, self.b = best_params      # restore the best checkpoint
        return self.history

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self.logits(X))
