"""Page 3 - inference: ask AlgoSathi and inspect its sources."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.rag_pipeline import AlgoSathiRAG  # noqa: E402
from src.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="Chat | AlgoSathi", page_icon="💬", layout="wide")
cfg = load_config("configs/default.yaml")

EXAMPLES = [
    "yo recursion ko base case kasari define garne hola?",
    "what is the time complexity of merge sort and why?",
    "binary search ko upper bound bhaneko k ho?",
    "dijkstra negative weight ma kina fail huncha?",
    "explain how a hash table handles collisions",
]


@st.cache_resource(show_spinner="Loading index, detector and generator...")
def get_rag():
    return AlgoSathiRAG(cfg)


st.title("💬 Ask AlgoSathi")

try:
    rag = get_rag()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

with st.sidebar:
    st.header("Retrieval controls")
    top_k = st.slider("Chunks to the generator", 1, 10,
                      cfg.get("retrieval.final_top_k", 5))
    use_dense = st.checkbox("Dense retrieval", True)
    use_bm25 = st.checkbox("BM25 lexical retrieval", True)
    rerank = st.checkbox("Cross-encoder re-ranking", True)
    override = st.selectbox("Language override "
                            "('my question contains Nepali' toggle)",
                            ["auto", "english", "romanized_nepali", "code_mixed"])
    st.caption(f"Generator backend: `{rag.generator.name}`")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3, 1])
question = col1.text_input("Your question (English, Nepali or a mix)",
                           EXAMPLES[0], label_visibility="visible")
example = col2.selectbox("Examples", EXAMPLES, index=0)
if col2.button("Use example"):
    question = example

if st.button("Ask", type="primary") and question.strip():
    response = rag.ask(question, top_k=top_k, use_dense=use_dense, use_bm25=use_bm25,
                       rerank=rerank,
                       force_language=None if override == "auto" else override)
    st.session_state.history.append(response)

for response in reversed(st.session_state.history[-5:]):
    st.divider()
    st.markdown(f"**You:** {response.query}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Register", response.query_type)
    c2.metric("Latency", f"{response.latency_s:.2f}s")
    c3.metric("Nepali tokens", f"{response.language_profile.get('nepali', 0):.0%}")
    c4.metric("Grounded", "yes" if response.grounded else "check")
    st.markdown(response.answer)

    with st.expander(f"Retrieved sources ({len(response.chunks)} chunks)"):
        table = pd.DataFrame([{
            "rank": c.get("rank"), "chunk_id": c["chunk_id"], "topic": c["topic"],
            "concept": c.get("concept_name"), "language": c["language"],
            "dense": round(c.get("dense_score", 0), 3),
            "bm25": round(c.get("bm25_score", 0), 3),
            "fusion": round(c.get("fusion_score", 0), 4),
            "rerank": round(c.get("rerank_score", 0), 3),
        } for c in response.chunks])
        st.dataframe(table, use_container_width=True)
        for chunk in response.chunks:
            with st.expander(f"[{chunk['chunk_id']}] {chunk.get('concept_name')}"):
                st.text(chunk["text"][:2500])
