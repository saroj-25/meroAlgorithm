"""Page 2 - train the language-ID model and rebuild the retrieval index."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import evaluate_query  # noqa: E402
from src.preprocessing.language_id import CharNGramLID  # noqa: E402
from src.preprocessing.lexicons import LABELS  # noqa: E402
from src.training.train_lid import classification_report, load_token_dataset, split_indices  # noqa: E402
from src.training.trainer import SoftmaxRegressionTrainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

st.set_page_config(page_title="Training | AlgoSathi", page_icon="🛠️", layout="wide")
cfg = load_config("configs/default.yaml")

st.title("🛠️ Training")

tab_lid, tab_index = st.tabs(["Language-ID classifier", "Retrieval index"])

# --------------------------------------------------------------------------- #
with tab_lid:
    st.markdown("Train the character n-gram code-switch classifier "
                "(Section 3.3). Runs in seconds on CPU.")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.slider("Epochs", 5, 100, cfg.get("language_id.epochs", 40), 5)
    lr = c2.select_slider("Learning rate", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
                          value=cfg.get("language_id.learning_rate", 0.5))
    batch = c3.select_slider("Batch size", [32, 64, 128, 256, 512],
                             value=cfg.get("language_id.batch_size", 128))
    l2 = c4.select_slider("L2", [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                          value=cfg.get("language_id.l2", 1e-4))

    if st.button("Train", type="primary"):
        rng = set_seed(cfg.get("project.seed", 42))
        tokens, y = load_token_dataset(cfg.path("paths.processed_dir") / "lid_dataset.jsonl")
        model = CharNGramLID(ngram_range=cfg.get("language_id.char_ngram_range", [1, 4]),
                             max_features=cfg.get("language_id.max_features", 30000))
        with st.spinner("Vectorising character n-grams..."):
            X = model.fit_vectorizer(tokens).tocsr()
        train_idx, val_idx = split_indices(X.shape[0],
                                           cfg.get("language_id.val_fraction", 0.15), rng)
        counts = np.bincount(y[train_idx], minlength=len(LABELS)).astype(float)
        weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))

        trainer = SoftmaxRegressionTrainer(
            n_features=X.shape[1], n_classes=len(LABELS), learning_rate=lr, l2=l2,
            batch_size=batch, epochs=epochs,
            patience=cfg.get("language_id.early_stopping_patience", 6),
            class_weights=weights, seed=cfg.get("project.seed", 42))

        progress = st.progress(0.0, "training...")
        chart = st.empty()
        history_rows = []
        for epoch in range(1, epochs + 1):
            trainer.epochs = 1                       # one epoch per UI update
            trainer.fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx], verbose=False)
            record = trainer.history.as_records()[-1]
            record["epoch"] = epoch
            history_rows.append(record)
            progress.progress(epoch / epochs, f"epoch {epoch}/{epochs}")
            df = pd.DataFrame(history_rows)
            chart.plotly_chart(
                px.line(df.melt(id_vars="epoch",
                                value_vars=["train_loss", "val_loss", "val_macro_f1"]),
                        x="epoch", y="value", color="variable",
                        title="Training curves"),
                use_container_width=True)

        model.W, model.b = trainer.W, trainer.b
        model.history = history_rows
        model.save(cfg.path("paths.checkpoints_dir") / "lid_model.pkl")
        report = classification_report(y[val_idx], trainer.predict(X[val_idx]))
        st.success(f"Saved checkpoint. Held-out macro-F1 = {report['macro_f1']:.3f}")
        st.json(report)

    st.divider()
    st.subheader("Try the detector")
    probe = st.text_input("A question in any register",
                          "yo binary search ko lower bound bhaneko k ho")
    if probe:
        from src.preprocessing.language_id import load_lid

        detector = load_lid(cfg.path("paths.checkpoints_dir") / "lid_model.pkl")
        from src.preprocessing.lexicons import tokenize

        tokens = tokenize(probe)
        st.write(f"**Query type:** `{detector.classify_query(probe)}`")
        st.dataframe(pd.DataFrame({"token": tokens,
                                   "label": detector.predict_tokens(tokens)}).T,
                     use_container_width=True)
        st.json({k: round(v, 3) for k, v in detector.language_profile(probe).items()})

# --------------------------------------------------------------------------- #
with tab_index:
    st.markdown("Encode the knowledge base and build the dense + BM25 indexes "
                "(offline phase of Figure 2). Needed after changing the corpus.")
    meta_path = cfg.path("paths.checkpoints_dir") / "index" / "index_meta.json"
    if meta_path.exists():
        from src.utils.io import read_json

        st.json(read_json(meta_path))
    if st.button("Rebuild index", type="primary"):
        from src.retrieval.index_builder import build_index

        with st.spinner("Encoding corpus (this can take a minute)..."):
            meta = build_index(cfg)
        st.success("Index rebuilt")
        st.json(meta)
        st.cache_resource.clear()
