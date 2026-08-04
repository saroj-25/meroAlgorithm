"""Hyper-parameter sweep for the retrieval stack.

"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from src.evaluation.evaluate_retrieval import run_ablation
from src.models.bm25 import BM25Okapi
from src.retrieval.hybrid import HybridRetriever
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.logging_utils import get_logger
from src.utils.tracking import RunTracker

log = get_logger("experiments.hparams")

# parameter name -> (config key, needs BM25 rebuild)
SWEEPABLE = {
    "rrf_k": ("retrieval.rrf_k", False),
    "dense_top_k": ("retrieval.dense_top_k", False),
    "bm25_top_k": ("retrieval.bm25_top_k", False),
    "rerank_top_k": ("retrieval.rerank_top_k", False),
    "bm25_weight": ("retrieval.fusion_weights.bm25", False),
    "register_boost": ("retrieval.register_boost", False),
    "bm25_k1": ("retrieval.bm25.k1", True),
    "bm25_b": ("retrieval.bm25.b", True),
}

BEST_CONFIG = [{"name": "hybrid_rerank", "use_dense": True, "use_bm25": True,
                "rerank": True}]


def sweep(cfg, param: str, values: List[float], retriever: HybridRetriever) -> pd.DataFrame:
    key, needs_bm25 = SWEEPABLE[param]
    rows = []
    for value in values:
        trial_cfg = cfg.copy_with(**{key.replace(".", "__"): value})
        retriever.cfg = trial_cfg
        if needs_bm25:
            retriever.bm25 = BM25Okapi([c["text"] for c in retriever.chunks],
                                       k1=trial_cfg.get("retrieval.bm25.k1"),
                                       b=trial_cfg.get("retrieval.bm25.b"))
        summary, _ = run_ablation(trial_cfg, retriever=retriever,
                                  retriever_configs=BEST_CONFIG)
        for _, row in summary.iterrows():
            rows.append({"param": param, "value": value,
                         "query_type": row["query_type"], "P@5": row["P@5"],
                         "MRR": row["MRR"], "R@10": row["R@10"],
                         "median_latency_ms": row["median_latency_ms"]})
        log.info("%s = %-6s -> P@5(ALL) = %.3f", param, value,
                 float(summary[summary.query_type == "ALL"]["P@5"].iloc[0]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hyper-parameter sweep")
    parser.add_argument("--config", default="configs/hparams.yaml")
    parser.add_argument("--param", default=None, choices=list(SWEEPABLE),
                        help="sweep a single parameter instead of the config grid")
    parser.add_argument("--values", default=None, help="comma-separated values")
    args = parser.parse_args()

    cfg = load_config(args.config)
    retriever = HybridRetriever.from_checkpoint(cfg)
    queries = read_jsonl(cfg.path("paths.processed_dir") / "eval_queries.jsonl")
    log.info("Sweeping over %d evaluation queries", len(queries))

    if args.param:
        grid = {args.param: [float(v) if "." in v else int(v)
                             for v in args.values.split(",")]}
    else:
        sweep_cfg = cfg.get("sweep") or {}
        grid = {k: v for k, v in sweep_cfg.items() if k in SWEEPABLE}
        grid.setdefault("rrf_k", [10, 30, 60, 100])
        grid.setdefault("bm25_weight", [0.0, 0.2, 0.4, 0.7, 1.0])

    frames = []
    with RunTracker("hyperparameter_search", cfg,
                    backend=cfg.get("tracking.backend", "auto")) as run:
        for param, values in grid.items():
            df = sweep(cfg, param, values, retriever)
            frames.append(df)
            try:
                from src.visualization.plots import plot_hparam_sweep

                fig = plot_hparam_sweep(df[df.query_type != "ALL"],
                                        cfg.path("paths.figures_dir") / f"hparam_{param}.png",
                                        x="value")
                run.log_artifact(fig)
            except Exception as exc:
                log.warning("Plot failed for %s: %s", param, exc)

        result = pd.concat(frames, ignore_index=True)
        out = cfg.path("paths.tables_dir") / "hparam_sweep.csv"
        result.to_csv(out, index=False)
        run.log_artifact(out)

        best = (result[result.query_type == "ALL"]
                .sort_values("P@5", ascending=False).head(10))
        print("\n=== Best settings by overall P@5 ===")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
