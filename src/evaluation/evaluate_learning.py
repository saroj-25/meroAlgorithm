"""Reproduce the learning-outcome analysis: Tables 5, 6 and 7.

    python -m src.evaluation.evaluate_learning --config configs/default.yaml
    python -m src.evaluation.evaluate_learning --scores path/to/real_scores.csv

A real CSV needs the columns ``student_id, group, pre_test, post_test`` where
``group`` is ``experimental`` or ``control``.  Everything downstream - the
t-test, effect size, ANCOVA, assumption checks, power - is computed from that
file alone, so the same pipeline that reproduces the paper analyses your data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import Config, load_config
from ..utils.io import write_json
from ..utils.logging_utils import get_logger
from ..utils.tracking import RunTracker
from .stats_tests import (ancova, assumption_checks, correlation, cronbach_alpha,
                          independent_t_test, power_two_sample_t,
                          required_n_per_group)

log = get_logger("evaluation.learning")


def analyse_scores(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    scores = scores.copy()
    if "gain" not in scores:
        scores["gain"] = scores["post_test"] - scores["pre_test"]

    exp = scores[scores.group == "experimental"]
    ctrl = scores[scores.group == "control"]

    baseline = independent_t_test(exp["pre_test"], ctrl["pre_test"], alpha)
    gain_test = independent_t_test(exp["gain"], ctrl["gain"], alpha)
    post_test = independent_t_test(exp["post_test"], ctrl["post_test"], alpha)

    adjusted = ancova(scores["post_test"],
                      (scores.group == "experimental").astype(int),
                      scores["pre_test"], alpha)

    checks = assumption_checks(exp["pre_test"], ctrl["pre_test"],
                               exp["gain"], ctrl["gain"], alpha)

    n_per_group = min(len(exp), len(ctrl))
    power = {
        "a_priori_power_d1.00": round(power_two_sample_t(1.00, n_per_group, alpha), 3),
        "observed_d": round(gain_test.cohens_d, 3),
        "observed_power_post_hoc": round(
            power_two_sample_t(abs(gain_test.cohens_d), n_per_group, alpha), 3),
        "n_needed_for_power_0.80_at_d1.00": required_n_per_group(1.00, 0.80, alpha),
        "note": ("Post-hoc power computed from the observed effect is a "
                 "restatement of the p-value; only the a priori figure informs "
                 "design adequacy."),
    }

    return {
        "descriptives": {
            group: {col: {"mean": round(float(sub[col].mean()), 2),
                          "sd": round(float(sub[col].std(ddof=1)), 2),
                          "n": int(sub[col].count())}
                    for col in ("pre_test", "post_test", "gain")}
            for group, sub in scores.groupby("group")},
        "baseline_equivalence": baseline.as_dict(),
        "gain_t_test": gain_test.as_dict(),
        "post_test_t_test": post_test.as_dict(),
        "ancova": adjusted,
        "assumption_checks": checks,
        "power": power,
    }


def table6(scores: pd.DataFrame, results: dict) -> pd.DataFrame:
    exp = scores[scores.group == "experimental"]
    ctrl = scores[scores.group == "control"]
    gain, anc = results["gain_t_test"], results["ancova"]
    rows = [
        {"group": f"Experimental (n = {len(exp)})",
         "pre_test_M_SD": f"{exp.pre_test.mean():.1f} ({exp.pre_test.std():.1f})",
         "post_test_M_SD": f"{exp.post_test.mean():.1f} ({exp.post_test.std():.1f})",
         "gain_M_SD": f"{exp.gain.mean():.1f} ({exp.gain.std():.1f})", "p": "-"},
        {"group": f"Control (n = {len(ctrl)})",
         "pre_test_M_SD": f"{ctrl.pre_test.mean():.1f} ({ctrl.pre_test.std():.1f})",
         "post_test_M_SD": f"{ctrl.post_test.mean():.1f} ({ctrl.post_test.std():.1f})",
         "gain_M_SD": f"{ctrl.gain.mean():.1f} ({ctrl.gain.std():.1f})", "p": "-"},
        {"group": "ANCOVA-adjusted between-group diff. (post-test)",
         "pre_test_M_SD": "-",
         "post_test_M_SD": (f"{anc['adjusted_difference']:.1f} "
                            f"(95% CI: {anc['ci_low']:.1f}-{anc['ci_high']:.1f})"),
         "gain_M_SD": "-", "p": f"{anc['p_value']:.2e}"},
        {"group": "Between-group diff. (gain, t-test)",
         "pre_test_M_SD": "-", "post_test_M_SD": "-",
         "gain_M_SD": (f"{gain['mean_diff']:.1f} "
                       f"(95% CI: {gain['ci_low']:.1f}-{gain['ci_high']:.1f})"),
         "p": f"{gain['p_value']:.2e}"},
        {"group": "Cohen's d", "pre_test_M_SD": "-",
         "post_test_M_SD": f"{results['post_test_t_test']['cohens_d']:.2f} (post-test)",
         "gain_M_SD": f"{gain['cohens_d']:.2f} (gain)", "p": "-"},
    ]
    return pd.DataFrame(rows)


def satisfaction_tables(responses: pd.DataFrame) -> tuple:
    """Sub-scale means (Table 7) plus per-item means and Cronbach's alpha."""
    subscales = (responses.groupby("subscale")["response"]
                 .agg(items=lambda s: s.index.size // responses.student_id.nunique(),
                      mean="mean", sd="std").reset_index().round(3))
    overall = pd.DataFrame([{
        "subscale": "overall",
        "items": responses.item_id.nunique(),
        "mean": round(responses["response"].mean(), 3),
        "sd": round(responses.groupby("student_id")["response"].mean().std(), 3),
    }])
    subscales = pd.concat([subscales, overall], ignore_index=True)

    items = (responses.groupby(["item_id", "item_text", "subscale"])["response"]
             .agg(["mean", "std"]).reset_index()
             .sort_values("mean", ascending=False).round(3))

    matrix = responses.pivot_table(index="student_id", columns="item_id",
                                   values="response").to_numpy()
    return subscales, items, round(cronbach_alpha(matrix), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Tables 5-7")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scores", default=None,
                        help="CSV with student_id, group, pre_test, post_test")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    processed = cfg.path("paths.processed_dir")
    tables = cfg.path("paths.tables_dir")
    figures = cfg.path("paths.figures_dir")
    alpha = cfg.get("evaluation.alpha", 0.05)

    scores_path = Path(args.scores) if args.scores else processed / "study_scores.csv"
    scores = pd.read_csv(scores_path)
    scores["gain"] = scores["post_test"] - scores["pre_test"]
    log.info("Loaded %d students from %s", len(scores), scores_path)

    with RunTracker("evaluate_learning", cfg,
                    backend=cfg.get("tracking.backend", "auto")) as run:
        results = analyse_scores(scores, alpha)

        pd.DataFrame(results["assumption_checks"]).to_csv(
            tables / "table5_assumptions.csv", index=False)
        table6(scores, results).to_csv(tables / "table6_learning_outcomes.csv", index=False)
        write_json(tables / "learning_analysis.json", results)

        run.log_metrics({
            "gain_diff": results["gain_t_test"]["mean_diff"],
            "gain_t": results["gain_t_test"]["t"],
            "cohens_d": results["gain_t_test"]["cohens_d"],
            "ancova_adjusted_diff": results["ancova"]["adjusted_difference"],
            "a_priori_power": results["power"]["a_priori_power_d1.00"],
        })

        # Satisfaction (Table 7)
        sat_path = processed / "satisfaction_responses.csv"
        if sat_path.exists():
            responses = pd.read_csv(sat_path)
            subscales, items, alpha_c = satisfaction_tables(responses)
            subscales.to_csv(tables / "table7_satisfaction.csv", index=False)
            items.to_csv(tables / "satisfaction_items.csv", index=False)
            write_json(tables / "satisfaction_meta.json", {"cronbach_alpha": alpha_c})
            run.log_metrics({"satisfaction_overall": float(subscales["mean"].iloc[-1]),
                             "cronbach_alpha": alpha_c})

        # Engagement correlation (Section 6.6)
        if "messages_sent" in scores.columns:
            exp = scores[scores.group == "experimental"]
            engagement = correlation(exp["messages_sent"], exp["gain"])
            write_json(tables / "engagement_correlation.json", engagement)
            run.log_metrics({"engagement_r": engagement["pearson_r"]})

        # Figures
        try:
            from ..visualization.plots import (plot_learning_outcomes, plot_query_mix,
                                               plot_satisfaction, plot_topic_gains)

            run.log_artifact(plot_learning_outcomes(scores, figures / "learning_outcomes.png"))
            if sat_path.exists():
                run.log_artifact(plot_satisfaction(subscales[subscales.subscale != "overall"],
                                                   figures / "satisfaction.png"))
            logs_path = processed / "interaction_logs.csv"
            if logs_path.exists():
                run.log_artifact(plot_query_mix(pd.read_csv(logs_path),
                                                figures / "query_mix.png"))
            topics_path = processed / "topic_gains.csv"
            if topics_path.exists():
                run.log_artifact(plot_topic_gains(pd.read_csv(topics_path),
                                                  figures / "topic_gains.png"))
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    gain, anc, power = results["gain_t_test"], results["ancova"], results["power"]
    print("\n=== Learning outcomes (reproduction) ===")
    print(f"gain difference : {gain['mean_diff']:.1f} pp "
          f"(95% CI {gain['ci_low']:.1f}-{gain['ci_high']:.1f})")
    print(f"t({gain['df']})          : {gain['t']:.2f}   p = {gain['p_value']:.2e}   "
          f"d = {gain['cohens_d']:.2f}")
    print(f"ANCOVA adjusted : {anc['adjusted_difference']:.1f} "
          f"(95% CI {anc['ci_low']:.1f}-{anc['ci_high']:.1f}), p = {anc['p_value']:.2e}, "
          f"partial eta^2 = {anc['partial_eta_squared']}")
    print(f"slopes homogeneous: F = {anc['slopes_interaction_F']}, "
          f"p = {anc['slopes_interaction_p']}")
    print(f"power (a priori, d=1.00): {power['a_priori_power_d1.00']}   "
          f"(post-hoc at observed d: {power['observed_power_post_hoc']})")
    print("\n=== Table 5: assumption checks ===")
    print(pd.DataFrame(results["assumption_checks"]).to_string(index=False))
    print("\nTables written to results/tables/")


if __name__ == "__main__":
    main()
