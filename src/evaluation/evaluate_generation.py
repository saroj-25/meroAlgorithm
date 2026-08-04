"""Evaluate generation quality by query language (Table 4 reproduction).

Reference answers are built from the *gold* chunk of each evaluation query, so
the metric asks: "did the system say what the course material says about this
concept, in the learner's register, with a valid citation?"

Run::

    python -m src.evaluation.evaluate_generation --config configs/default.yaml --n 60
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from ..inference.rag_pipeline import AlgoSathiRAG
from ..utils.config import Config, load_config
from ..utils.io import read_jsonl, write_json
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..utils.tracking import RunTracker
from .generation_metrics import (bertscore_f1, grounding_rate, register_match,
                                 rouge_l)

log = get_logger("evaluation.generation")


def build_reference(chunks: List[dict], concept: str) -> str:
    """Reference answer = the instructor content for that concept."""
    texts = [c["text"] for c in chunks if c["concept"] == concept]
    return " ".join(texts[:2])


def evaluate_generation(cfg: Config, n_per_language: int = 40,
                        rag: AlgoSathiRAG | None = None):
    set_seed(cfg.get("project.seed", 42))
    rag = rag or AlgoSathiRAG(cfg)
    processed = cfg.path("paths.processed_dir")
    queries = read_jsonl(processed / "eval_queries.jsonl")
    kb = read_jsonl(processed / "knowledge_base.jsonl")

    by_language: Dict[str, List[dict]] = {}
    for query in queries:
        by_language.setdefault(query["query_type"], []).append(query)

    rows: List[dict] = []
    for language, group in by_language.items():
        for query in group[:n_per_language]:
            response = rag.ask(query["query"])
            reference = build_reference(kb, query["concept"])
            retrieved_ids = {c["chunk_id"] for c in response.chunks}
            valid_citations = [c for c in response.citations if c in retrieved_ids]
            rows.append({
                "query_id": query["query_id"], "query": query["query"],
                "query_type": language, "concept": query["concept"],
                "answer": response.answer,
                "reference": reference,
                "rougeL_f1": rouge_l(response.answer, reference),
                "grounding_rate": grounding_rate(response.answer, response.chunks),
                "citation_rate": 1.0 if valid_citations else 0.0,
                "invalid_citations": len(response.citations) - len(valid_citations),
                "register_match": register_match(response.language_profile,
                                                 response.answer, rag.detector),
                # graded view of the same idea: how Nepali is the reply, relative
                # to how Nepali the question was?
                "nepali_ratio_query": response.language_profile.get("nepali", 0.0),
                "nepali_ratio_answer": rag.detector.language_profile(
                    response.answer).get("nepali", 0.0),
                "retrieval_hit": float(any(
                    c["chunk_id"] in set(query["relevant_chunk_ids"])
                    for c in response.chunks)),
                "latency_s": response.latency_s,
            })

    df = pd.DataFrame(rows)
    bert = bertscore_f1(df["answer"].tolist(), df["reference"].tolist())
    df[bert["metric"]] = bert["values"]

    metric_cols = [c for c in ["rougeL_f1", bert["metric"], "grounding_rate",
                               "citation_rate", "register_match",
                               "nepali_ratio_query", "nepali_ratio_answer",
                               "retrieval_hit", "latency_s"] if c in df.columns]
    summary = (df.groupby("query_type")[metric_cols].mean().round(4).reset_index())
    overall = df[metric_cols].mean().round(4).to_dict()
    overall["query_type"] = "Overall"
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary, df, bert["metric"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation (Table 4)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=40,
                        help="queries per language (LLM backends are slow)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    with RunTracker("evaluate_generation", cfg,
                    backend=cfg.get("tracking.backend", "auto")) as run:
        summary, per_query, semantic_metric = evaluate_generation(cfg, args.n)
        tables = cfg.path("paths.tables_dir")
        summary.to_csv(tables / "table4_generation.csv", index=False)
        per_query.drop(columns=["reference"]).to_csv(
            tables / "generation_per_query.csv", index=False)
        write_json(tables / "generation_meta.json",
                   {"semantic_metric": semantic_metric,
                    "note": ("token_f1_proxy means bert-score is not installed; "
                             "install requirements-full.txt for real BERTScore")})
        for _, row in summary.iterrows():
            run.log_metrics({f"{row['query_type']}/{m}": row[m]
                             for m in summary.columns if m != "query_type"})

        try:
            from ..visualization.plots import plot_generation_quality

            plot_df = summary.rename(columns={semantic_metric: "bertscore_f1"})
            run.log_artifact(plot_generation_quality(
                plot_df, cfg.path("paths.figures_dir") / "generation_quality.png"))
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    print("\n=== Table 4 (reproduction) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
