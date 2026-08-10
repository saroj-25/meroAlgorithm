# knowledge_base/build_index.py
# Run this ONCE to build the FAISS index from your raw documents
# Usage: python knowledge_base/build_index.py

import os
import sys
import faiss
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.chunker  import chunk_text_files
from knowledge_base.embedder import DSAEmbedder


def build_index(
    raw_dir      = "data/raw",
    chunks_path  = "data/processed/chunks.json",
    index_dir    = "indexes"
):
    print("=" * 50)
    print("  MeroAlgorithm — Building Knowledge Index")
    print("=" * 50)

    # ── Step 1: Chunk documents ───────────────────────
    print("\n[Step 1/3] Chunking documents...")
    chunks = chunk_text_files(raw_dir, chunks_path)

    if not chunks:
        print("ERROR: No chunks created. Add .txt files to data/raw/ first.")
        return

    # ── Step 2: Embed chunks ──────────────────────────
    print("\n[Step 2/3] Creating embeddings...")
    embedder   = DSAEmbedder()
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    # ── Step 3: Build FAISS HNSW index ───────────────
    print("\n[Step 3/3] Building FAISS HNSW index...")
    dim   = embeddings.shape[1]           # 768 for mpnet
    index = faiss.IndexHNSWFlat(dim, 32)  # M=32 neighbours
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    # ── Save index and chunks ─────────────────────────
    os.makedirs(index_dir, exist_ok=True)
    index_path  = os.path.join(index_dir, "dsa.index")
    chunks_pkl  = os.path.join(index_dir, "chunks.pkl")

    faiss.write_index(index, index_path)
    with open(chunks_pkl, "wb") as f:
        pickle.dump(chunks, f)

    print("\n" + "=" * 50)
    print(f"  Done! {index.ntotal} vectors indexed.")
    print(f"  Index  → {index_path}")
    print(f"  Chunks → {chunks_pkl}")
    print("=" * 50)
    print("\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    build_index()
