"""Retrieval evaluation and the ablation grid of Table 3.

Runs every retriever configuration over every query-language slice of the
300-query annotated set and writes:

    results/tables/retrieval_ablation.csv     full grid (all metrics)
    results/tables/table3_retrieval.csv       paper-shaped table (P@5, MRR, R@10)
    results/tables/retrieval_per_query.csv    per-query scores, for error analysis
    results/figures/retrieval_ablation_p5.png

Run::

    python -m src.evaluation.evaluate_retrieval --config configs/ablation.yaml
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import pandas as pd

from ..preprocessing.language_id import load_lid
from ..preprocessing.query_pipeline import preprocess_query
from ..retrieval.hybrid import HybridRetriever
from ..utils.config import Config, load_config
from ..utils.io import read_jsonl
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..utils.tracking import RunTracker
from .retrieval_metrics import aggregate, evaluate_query

log = get_logger("evaluation.retrieval")

DEFAULT_CONFIGS = [
    {"name": "dense_only", "use_dense": True, "use_bm25": False, "rerank": False},
    {"name": "bm25_only", "use_dense": False, "use_bm25": True, "rerank": False},
    {"name": "hybrid_no_rerank", "use_dense": True, "use_bm25": True, "rerank": False},
    {"name": "hybrid_rerank", "use_dense": True, "use_bm25": True, "rerank": True},
]


def run_ablation(cfg: Config, retriever: HybridRetriever | None = None,
                 queries: List[dict] | None = None,
                 retriever_configs: List[dict] | None = None):
    """Return (summary_df, per_query_df) over the retriever x language grid."""
    set_seed(cfg.get("project.seed", 42))
    retriever = retriever or HybridRetriever.from_checkpoint(cfg)
    queries = queries or read_jsonl(cfg.path("paths.processed_dir") / "eval_queries.jsonl")
    retriever_configs = retriever_configs or cfg.get("ablation.retrievers") or DEFAULT_CONFIGS

    detector = load_lid(cfg.path("paths.checkpoints_dir") / "lid_model.pkl")
    ks = cfg.get("evaluation.retrieval_k", [1, 3, 5, 10])
    max_k = max(ks)

    # Pre-process each query once; the branch depends only on the query, not the
    # retriever configuration under test.
    processed = {q["query_id"]: preprocess_query(q["query"], detector, cfg) for q in queries}

    rows: List[Dict] = []
    per_query: List[Dict] = []

    for rconf in retriever_configs:
        for query in queries:
            pq = processed[query["query_id"]]
            t0 = time.perf_counter()
            hits = retriever.retrieve(pq, top_k=max_k,
                                      use_dense=rconf["use_dense"],
                                      use_bm25=rconf["use_bm25"],
                                      rerank=rconf["rerank"],
                                      dense_weight=rconf.get("dense_weight"),
                                      bm25_weight=rconf.get("bm25_weight"))
            latency_ms = (time.perf_counter() - t0) * 1000
            retrieved_ids = [h["chunk_id"] for h in hits]
            metrics = evaluate_query(retrieved_ids, query["relevant_chunk_ids"], ks)
            per_query.append({
                "retriever": rconf["name"], "query_id": query["query_id"],
                "query": query["query"], "query_type": query["query_type"],
                "concept": query["concept"], "topic": query["topic"],
                "detected_type": pq.query_type, "expanded": pq.expanded,
                "latency_ms": round(latency_ms, 2),
                "top1_chunk": retrieved_ids[0] if retrieved_ids else "",
                **metrics,
            })

    per_query_df = pd.DataFrame(per_query)
    for (name, qtype), group in per_query_df.groupby(["retriever", "query_type"]):
        agg = aggregate(group[[c for c in group.columns
                               if c.startswith(("P@", "R@", "nDCG@")) or c == "MRR"]]
                        .to_dict("records"))
        rows.append({"retriever": name, "query_type": qtype, "n_queries": len(group),
                     "median_latency_ms": float(group["latency_ms"].median()),
                     **{k: round(v, 4) for k, v in agg.items()}})

    # "overall" row per retriever, matching the paper's headline P@5 numbers
    for name, group in per_query_df.groupby("retriever"):
        agg = aggregate(group[[c for c in group.columns
                               if c.startswith(("P@", "R@", "nDCG@")) or c == "MRR"]]
                        .to_dict("records"))
        rows.append({"retriever": name, "query_type": "ALL", "n_queries": len(group),
                     "median_latency_ms": float(group["latency_ms"].median()),
                     **{k: round(v, 4) for k, v in agg.items()}})

    summary_df = pd.DataFrame(rows).sort_values(["query_type", "retriever"])
    return summary_df, per_query_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (Table 3)")
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    with RunTracker("evaluate_retrieval", cfg,
                    backend=cfg.get("tracking.backend", "auto")) as run:
        summary, per_query = run_ablation(cfg)
        tables = cfg.path("paths.tables_dir")
        summary.to_csv(tables / "retrieval_ablation.csv", index=False)
        per_query.to_csv(tables / "retrieval_per_query.csv", index=False)

        table3 = (summary[summary.query_type != "ALL"]
                  [["query_type", "retriever", "P@5", "MRR", "R@10"]]
                  .sort_values(["query_type", "retriever"]))
        table3.to_csv(tables / "table3_retrieval.csv", index=False)

        for _, row in summary.iterrows():
            run.log_metrics({f"{row['retriever']}/{row['query_type']}/P@5": row["P@5"],
                             f"{row['retriever']}/{row['query_type']}/MRR": row["MRR"],
                             f"{row['retriever']}/{row['query_type']}/R@10": row["R@10"]})

        try:
            from ..visualization.plots import plot_retrieval_ablation

            fig = plot_retrieval_ablation(
                summary[summary.query_type != "ALL"],
                cfg.path("paths.figures_dir") / "retrieval_ablation_p5.png", "P@5")
            run.log_artifact(fig)
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    print("\n=== Table 3 (reproduction) ===")
    print(table3.to_string(index=False))
    print("\nSaved to results/tables/")


if __name__ == "__main__":
    main()
