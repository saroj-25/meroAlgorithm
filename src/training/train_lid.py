from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..preprocessing.language_id import CharNGramLID
from ..preprocessing.lexicons import LABELS
from ..utils.config import Config, load_config
from ..utils.io import read_jsonl, write_json
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..utils.tracking import RunTracker
from .trainer import SoftmaxRegressionTrainer, macro_f1

log = get_logger("training.lid")


def load_token_dataset(path) -> Tuple[List[str], np.ndarray]:
    """Flatten the sentence-level JSONL into (token, label-index) pairs."""
    rows = read_jsonl(path)
    tokens: List[str] = []
    labels: List[int] = []
    for row in rows:
        for token, label in zip(row["tokens"], row["labels"]):
            tokens.append(token)
            labels.append(LABELS.index(label))
    return tokens, np.array(labels, dtype=np.int64)


def split_indices(n: int, val_fraction: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return order[n_val:], order[:n_val]


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = {}
    for i, label in enumerate(LABELS):
        tp = float(np.sum((y_pred == i) & (y_true == i)))
        fp = float(np.sum((y_pred == i) & (y_true != i)))
        fn = float(np.sum((y_pred != i) & (y_true == i)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                         "f1": round(f1, 4), "support": int(np.sum(y_true == i))}
    report["macro_f1"] = round(macro_f1(y_true, y_pred, len(LABELS)), 4)
    report["accuracy"] = round(float(np.mean(y_true == y_pred)), 4)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the language-ID classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    rng = set_seed(cfg.get("project.seed", 42))

    tokens, y = load_token_dataset(cfg.path("paths.processed_dir") / "lid_dataset.jsonl")
    log.info("Loaded %d labelled tokens (%s)", len(tokens),
             {lab: int(np.sum(y == i)) for i, lab in enumerate(LABELS)})

    model = CharNGramLID(ngram_range=cfg.get("language_id.char_ngram_range", [1, 4]),
                         max_features=cfg.get("language_id.max_features", 30000))
    X = model.fit_vectorizer(tokens).tocsr()

    train_idx, val_idx = split_indices(X.shape[0],
                                       cfg.get("language_id.val_fraction", 0.15), rng)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Class weights counter the natural imbalance between English and Nepali tokens.
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(float)
    class_weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))

    trainer = SoftmaxRegressionTrainer(
        n_features=X.shape[1], n_classes=len(LABELS),
        learning_rate=args.lr or cfg.get("language_id.learning_rate", 0.5),
        l2=cfg.get("language_id.l2", 1e-4),
        batch_size=cfg.get("language_id.batch_size", 128),
        epochs=args.epochs or cfg.get("language_id.epochs", 40),
        patience=cfg.get("language_id.early_stopping_patience", 6),
        class_weights=class_weights, seed=cfg.get("project.seed", 42))

    with RunTracker("train_lid", cfg, backend=cfg.get("tracking.backend", "auto")) as run:
        history = trainer.fit(X_train, y_train, X_val, y_val)
        for record in history.as_records():
            run.log_metrics({k: v for k, v in record.items() if k != "epoch"},
                            step=record["epoch"])

        model.W, model.b = trainer.W, trainer.b
        model.history = history.as_records()

        report = classification_report(y_val, trainer.predict(X_val))
        run.log_metrics({"val_macro_f1": report["macro_f1"],
                         "val_accuracy": report["accuracy"],
                         "n_epochs_run": len(history.epochs)})

        ckpt = model.save(cfg.path("paths.checkpoints_dir") / "lid_model.pkl")
        tables = cfg.path("paths.tables_dir")
        pd.DataFrame(history.as_records()).to_csv(tables / "lid_history.csv", index=False)
        write_json(tables / "lid_report.json", report)
        run.log_artifact(ckpt)

    log.info("Held-out report: %s", report)

    # Training curves
    try:
        from ..visualization.plots import plot_training_curves

        fig_path = plot_training_curves(
            history.as_records(), cfg.path("paths.figures_dir") / "lid_training_curves.png")
        log.info("Curves -> %s", fig_path)
    except Exception as exc:  # plotting must never break training
        log.warning("Could not plot training curves: %s", exc)


if __name__ == "__main__":
    main()
