from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import List

from ..models.bm25 import BM25Okapi
from ..models.encoders import build_encoder
from ..models.vector_store import build_vector_store
from ..utils.config import Config, load_config
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed

log = get_logger("retrieval.index")


def index_dir(cfg: Config) -> Path:
    return cfg.path("paths.checkpoints_dir") / "index"


def build_index(cfg: Config, chunks: List[dict] | None = None) -> dict:
    set_seed(cfg.get("project.seed", 42))
    processed = cfg.path("paths.processed_dir")
    if chunks is None:
        chunks = read_jsonl(processed / "knowledge_base.jsonl")
    texts = [c["text"] for c in chunks]
    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    encoder = build_encoder(cfg, corpus=texts)
    vectors = encoder.encode(texts, batch_size=cfg.get("encoder.batch_size", 32))
    t_encode = time.time() - t0

    store = build_vector_store(cfg, dim=int(vectors.shape[1]))
    store.add(vectors)

    t1 = time.time()
    bm25 = BM25Okapi(texts,
                     k1=cfg.get("retrieval.bm25.k1", 1.5),
                     b=cfg.get("retrieval.bm25.b", 0.75))
    t_bm25 = time.time() - t1

    encoder.save(out / "encoder.pkl")
    store.save(out / "vector_store.pkl")
    with open(out / "bm25.pkl", "wb") as fh:
        pickle.dump(bm25, fh)
    write_jsonl(out / "chunks.jsonl", chunks)

    meta = {
        "n_chunks": len(chunks),
        "encoder_backend": encoder.name,
        "encoder_dim": int(vectors.shape[1]),
        "vector_store_backend": store.backend,
        "bm25_k1": cfg.get("retrieval.bm25.k1"),
        "bm25_b": cfg.get("retrieval.bm25.b"),
        "encode_seconds": round(t_encode, 2),
        "bm25_seconds": round(t_bm25, 2),
        "vocabulary_size": len(bm25.postings),
        "avg_doc_len": round(bm25.avgdl, 1),
    }
    write_json(out / "index_meta.json", meta)
    log.info("Index built: %s", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    build_index(load_config(args.config))


if __name__ == "__main__":
    main()
