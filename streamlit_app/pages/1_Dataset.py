"""Page 1 - dataset information, generation and upload."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402

st.set_page_config(page_title="Dataset | AlgoSathi", page_icon="📊", layout="wide")
cfg = load_config("configs/default.yaml")
processed = cfg.path("paths.processed_dir")

st.title("📊 Dataset")

tab_kb, tab_queries, tab_study, tab_upload = st.tabs(
    ["Knowledge base", "Evaluation queries", "Study data", "Use your own material"])

# --------------------------------------------------------------------------- #
with tab_kb:
    kb_path = processed / "knowledge_base.jsonl"
    if not kb_path.exists():
        st.warning("No knowledge base yet.")
        if st.button("Build knowledge base (1,252 chunks)"):
            from src.data.build_knowledge_base import build_knowledge_base
            from src.utils.io import write_jsonl

            with st.spinner("Building..."):
                write_jsonl(kb_path, build_knowledge_base(cfg))
            st.rerun()
    else:
        chunks = pd.DataFrame(read_jsonl(kb_path))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", len(chunks))
        c2.metric("English", int((chunks.language == "en").sum()))
        c3.metric("Code-mixed", int((chunks.language == "cm").sum()))
        c4.metric("Mean tokens", round(chunks.n_tokens.mean(), 1))

        composition = (chunks.groupby(["topic", "language"]).size()
                       .reset_index(name="chunks"))
        st.plotly_chart(
            px.bar(composition, x="chunks", y="topic", color="language",
                   orientation="h", title="Table 1 — knowledge base composition",
                   height=520), use_container_width=True)

        st.plotly_chart(px.histogram(chunks, x="n_tokens", nbins=40, color="language",
                                     title="Chunk length distribution (target 250–450 tokens)"),
                        use_container_width=True)

        st.subheader("Inspect a chunk")
        topic = st.selectbox("Topic", sorted(chunks.topic.unique()))
        subset = chunks[chunks.topic == topic]
        chunk_id = st.selectbox("Chunk", subset.chunk_id.tolist())
        row = subset[subset.chunk_id == chunk_id].iloc[0]
        st.caption(f"concept={row.concept} · facet={row.facet} · source={row.source} "
                   f"· difficulty={row.difficulty} · {row.n_tokens} tokens")
        st.text(row.text)

# --------------------------------------------------------------------------- #
with tab_queries:
    q_path = processed / "eval_queries.jsonl"
    if not q_path.exists():
        st.warning("Run `python -m src.data.build_eval_sets` first.")
    else:
        queries = pd.DataFrame(read_jsonl(q_path))
        st.metric("Annotated queries", len(queries))
        st.plotly_chart(px.pie(queries, names="query_type",
                               title="300 annotated queries by language"),
                        use_container_width=True)
        qtype = st.selectbox("Filter by language", ["all"] + sorted(queries.query_type.unique()))
        view = queries if qtype == "all" else queries[queries.query_type == qtype]
        st.dataframe(view[["query_id", "query", "query_type", "concept", "topic",
                           "n_relevant"]], use_container_width=True, height=420)

# --------------------------------------------------------------------------- #
with tab_study:
    scores_path = processed / "study_scores.csv"
    if not scores_path.exists():
        st.warning("Run `python -m src.data.simulate_study_data` first.")
    else:
        st.info("Dataset")
        scores = pd.read_csv(scores_path)
        st.dataframe(scores.head(20), use_container_width=True)
        st.plotly_chart(px.box(scores, x="group", y="gain", points="all",
                               title="Learning gain by group"),
                        use_container_width=True)
        logs_path = processed / "interaction_logs.csv"
        if logs_path.exists():
            logs = pd.read_csv(logs_path)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(logs, names="query_type",
                                   title="Logged query language mix"),
                            use_container_width=True)
            weekly = logs.groupby(["week", "query_type"]).size().reset_index(name="queries")
            c2.plotly_chart(px.bar(weekly, x="week", y="queries", color="query_type",
                                   title="Engagement over the six-week deployment"),
                            use_container_width=True)

# --------------------------------------------------------------------------- #
with tab_upload:
    st.markdown("""
Upload your own lecture notes (`.md`, `.txt`) in this format.
    """)
    uploads = st.file_uploader("Course material", type=["md", "txt"],
                               accept_multiple_files=True)
    topic = st.text_input("Topic label for these files", "Custom Material")
    language = st.radio("Register of this material", ["en", "cm"], horizontal=True)
    if uploads and st.button("Ingest"):
        from src.preprocessing.chunking import chunk_text
        from src.utils.io import read_jsonl as _read, write_jsonl

        existing = _read(processed / "knowledge_base.jsonl") if \
            (processed / "knowledge_base.jsonl").exists() else []
        added = []
        for f_idx, upload in enumerate(uploads):
            text = upload.read().decode("utf-8", errors="ignore")
            for c_idx, piece in enumerate(chunk_text(
                    text,
                    min_tokens=cfg.get("knowledge_base.chunk_min_tokens", 250),
                    max_tokens=cfg.get("knowledge_base.chunk_max_tokens", 450),
                    overlap=cfg.get("knowledge_base.chunk_overlap_tokens", 50))):
                added.append({
                    "chunk_id": f"USR-{'EN' if language == 'en' else 'CM'}-{f_idx:02d}{c_idx:03d}",
                    "topic": topic, "concept": Path(upload.name).stem,
                    "concept_name": Path(upload.name).stem.replace("_", " ").title(),
                    "facet": "user_upload", "language": language,
                    "difficulty": "medium", "source": upload.name,
                    "text": piece, "n_tokens": len(piece.split())})
        write_jsonl(processed / "knowledge_base.jsonl", existing + added)
        st.success(f"Added {len(added)} chunks from {len(uploads)} file(s). "
                   "Now rebuild the index on the Training page.")
