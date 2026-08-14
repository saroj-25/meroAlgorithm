"""Run the whole reproduction end to end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger

log = get_logger("experiments.run_all")

STAGES = ["data", "index", "train", "retrieval", "generation", "learning", "report"]

# Headline numbers reported in the paper, used only for side-by-side reporting.
PAPER = {
    "P@5 code-mixed (hybrid+rerank)": 0.79,
    "P@5 English (hybrid+rerank)": 0.83,
    "P@5 Romanized Nepali (hybrid+rerank)": 0.71,
    "MRR code-mixed (hybrid+rerank)": 0.72,
    "gain difference (pp)": 11.7,
    "Cohen's d (gain)": 1.17,
    "t(50)": 4.21,
    "ANCOVA adjusted difference": 12.1,
    "satisfaction overall (1-5)": 4.31,
    "code-mixed share of queries": 0.431,
    "knowledge base chunks": 1252,
}


def stage_data(cfg_path: str) -> None:
    from src.data import build_eval_sets, build_knowledge_base, simulate_study_data

    for module in (build_knowledge_base, build_eval_sets, simulate_study_data):
        _run_module_main(module, cfg_path)


def _run_module_main(module, cfg_path: str) -> None:
    import sys

    argv = sys.argv
    sys.argv = [module.__name__, "--config", cfg_path]
    try:
        module.main()
    finally:
        sys.argv = argv


def write_report(cfg, cfg_path: str) -> Path:
    tables = cfg.path("paths.tables_dir")
    lines = ["# AlgoSathi  report", "",
             f"Config: `{cfg_path}`  |  generated: {time.strftime('%Y-%m-%d %H:%M')}", "",
             "All numbers below come from **this repository's synthetic corpus and "
             "study data**. They are a check that the pipeline computes what "
             "the paper describes, not independent confirmation of the paper's findings.",
             ""]

    ours = {}
    p_retrieval = tables / "table3_retrieval.csv"
    if p_retrieval.exists():
        df = pd.read_csv(p_retrieval)
        best = df[df.retriever.isin(["hybrid_rerank", "hybrid_weighted_rerank"])]
        lines += ["## Table 3 - retrieval", "", df.to_markdown(index=False), ""]
        for qtype, label in [("code_mixed", "code-mixed"), ("english", "English"),
                             ("romanized_nepali", "Romanized Nepali")]:
            sub = best[best.query_type == qtype]
            if len(sub):
                row = sub.sort_values("P@5", ascending=False).iloc[0]
                ours[f"P@5 {label} (hybrid+rerank)"] = float(row["P@5"])
                ours[f"MRR {label} (hybrid+rerank)"] = float(row["MRR"])

    p_gen = tables / "table4_generation.csv"
    if p_gen.exists():
        lines += ["## Table 4 - generation quality", "",
                  pd.read_csv(p_gen).to_markdown(index=False), ""]

    p_assume = tables / "table5_assumptions.csv"
    if p_assume.exists():
        lines += ["## Table 5 - assumption checks", "",
                  pd.read_csv(p_assume).to_markdown(index=False), ""]

    p_learn = tables / "table6_learning_outcomes.csv"
    if p_learn.exists():
        lines += ["## Table 6 - learning outcomes", "",
                  pd.read_csv(p_learn).to_markdown(index=False), ""]
    p_analysis = tables / "learning_analysis.json"
    if p_analysis.exists():
        analysis = read_json(p_analysis)
        ours["gain difference (pp)"] = analysis["gain_t_test"]["mean_diff"]
        ours["Cohen's d (gain)"] = analysis["gain_t_test"]["cohens_d"]
        ours["t(50)"] = analysis["gain_t_test"]["t"]
        ours["ANCOVA adjusted difference"] = analysis["ancova"]["adjusted_difference"]

    p_sat = tables / "table7_satisfaction.csv"
    if p_sat.exists():
        sat = pd.read_csv(p_sat)
        lines += ["## Table 7 - satisfaction", "", sat.to_markdown(index=False), ""]
        ours["satisfaction overall (1-5)"] = float(sat["mean"].iloc[-1])

    logs = cfg.path("paths.processed_dir") / "interaction_logs.csv"
    if logs.exists():
        share = pd.read_csv(logs)["query_type"].value_counts(normalize=True)
        ours["code-mixed share of queries"] = float(share.get("code_mixed", 0))

    kb_stats = cfg.path("paths.processed_dir") / "knowledge_base_stats.json"
    if kb_stats.exists():
        ours["knowledge base chunks"] = read_json(kb_stats)["total_chunks"]

    rows = [{"quantity": key, "paper": value,
             "this repo": round(ours[key], 4) if key in ours else "n/a",
             "delta": (round(ours[key] - value, 4) if key in ours else "")}
            for key, value in PAPER.items()]
    lines = lines[:4] + ["## Side by side with the paper", "",
                         pd.DataFrame(rows).to_markdown(index=False), ""] + lines[4:]

    out = cfg.path("paths.results_dir") / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    write_json(cfg.path("paths.results_dir") / "headline_metrics.json",
               {"paper": PAPER, "reproduction": ours})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full reproduction")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablation-config", default="configs/ablation.yaml")
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    parser.add_argument("--gen-n", type=int, default=25,
                        help="queries per language for the generation evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    started = time.time()

    if "data" not in args.skip:
        log.info("=== STAGE: data ===")
        stage_data(args.config)

    if "index" not in args.skip:
        log.info("=== STAGE: index ===")
        from src.retrieval.index_builder import build_index

        build_index(cfg)

    if "train" not in args.skip:
        log.info("=== STAGE: train (language ID) ===")
        from src.training import train_lid

        _run_module_main(train_lid, args.config)

    if "retrieval" not in args.skip:
        log.info("=== STAGE: retrieval ablation ===")
        from src.evaluation import evaluate_retrieval

        _run_module_main(evaluate_retrieval, args.ablation_config)

    if "generation" not in args.skip:
        log.info("=== STAGE: generation evaluation ===")
        import sys

        from src.evaluation import evaluate_generation

        argv = sys.argv
        sys.argv = ["evaluate_generation", "--config", args.config, "--n", str(args.gen_n)]
        try:
            evaluate_generation.main()
        finally:
            sys.argv = argv

    if "learning" not in args.skip:
        log.info("=== STAGE: learning outcomes ===")
        from src.evaluation import evaluate_learning

        _run_module_main(evaluate_learning, args.config)

    if "report" not in args.skip:
        log.info("=== STAGE: report ===")
        path = write_report(cfg, args.config)
        log.info("Report written to %s", path)

    log.info("Done in %.1f s. See results/REPORT.md", time.time() - started)


if __name__ == "__main__":
    main()
