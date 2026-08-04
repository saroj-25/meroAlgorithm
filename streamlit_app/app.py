"""AlgoSathi - Streamlit demo.

    streamlit run streamlit_app/app.py

Four pages (see streamlit_app/pages/):
    1. Dataset      - knowledge base, evaluation sets, study data, upload your own
    2. Training     - train the code-switch classifier and rebuild the index
    3. Chat         - ask AlgoSathi a question and inspect its retrieved sources
    4. Results      - retrieval, generation and learning-outcome results
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="AlgoSathi", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def get_config(path: str = "configs/default.yaml"):
    return load_config(path)


def sidebar_status(cfg) -> None:
    """Show which artefacts exist, so a new user knows what to run next."""
    st.sidebar.header("Pipeline status")
    processed = cfg.path("paths.processed_dir")
    checkpoints = cfg.path("paths.checkpoints_dir")
    items = {
        "Knowledge base": processed / "knowledge_base.jsonl",
        "Evaluation queries": processed / "eval_queries.jsonl",
        "Study data": processed / "study_scores.csv",
        "Retrieval index": checkpoints / "index" / "encoder.pkl",
        "Language-ID model": checkpoints / "lid_model.pkl",
    }
    for label, path in items.items():
        st.sidebar.write(("✅ " if path.exists() else "⬜ ") + label)
    if not all(p.exists() for p in items.values()):
        st.sidebar.info("Missing pieces? Run `make all` or use the Dataset "
                        "and Training pages.")


def main() -> None:
    cfg = get_config()
    st.title("🎓 AlgoSathi")
    st.caption("Retrieval-Augmented Generation for DSA learning in Romanized "
               "Nepali and English — a reproduction of Bhandari & Dhital (2026)")

    sidebar_status(cfg)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What this app is")
        st.markdown(
            """
This is a working implementation of the AlgoSathi pipeline:

1. **Preprocess** — Unicode normalization, token-level code-switch detection,
   optional Romanized → Devanagari expansion.
2. **Retrieve** — multilingual dense search + BM25, fused with reciprocal rank
   fusion, then cross-encoder re-ranking (50 → 20 → 5 chunks).
3. **Generate** — a grounded, cited, register-preserving answer with a Socratic
   check-back question.

Use the pages in the sidebar: **Dataset → Training → Chat → Results**.
            """)
        st.subheader("Honesty notice")
        st.warning(
            "The knowledge base and the 52-student study data shipped here are "
            "**synthetic**: the paper's corpus and human-subjects data are not "
            "public. Numbers reproduced in this app validate the *pipeline*, not "
            "the paper's empirical claims. Point the Dataset page at your own "
            "course notes and the Results page at a real scores CSV to do real work.")

    with right:
        st.subheader("Active configuration")
        st.json({
            "mode": cfg.get("project.mode"),
            "encoder": cfg.get("encoder.model_name"),
            "vector store": cfg.get("vector_store.backend"),
            "generator": cfg.get("generation.backend"),
            "top-k pipeline": f"{cfg.get('retrieval.dense_top_k')} → "
                              f"{cfg.get('retrieval.rerank_top_k')} → "
                              f"{cfg.get('retrieval.final_top_k')}",
            "RRF k": cfg.get("retrieval.rrf_k"),
            "seed": cfg.get("project.seed"),
        })


if __name__ == "__main__":
    main()
