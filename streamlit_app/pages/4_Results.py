"""Page 4 - results: retrieval, generation, learning outcomes, experiment runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_learning import analyse_scores, satisfaction_tables  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_json  # noqa: E402
from src.utils.tracking import load_runs  # noqa: E402

st.set_page_config(page_title="Results | AlgoSathi", page_icon="📈", layout="wide")
cfg = load_config("configs/default.yaml")
tables = cfg.path("paths.tables_dir")
figures = cfg.path("paths.figures_dir")

st.title("📈 Results")

tab_r, tab_g, tab_l, tab_runs = st.tabs(
    ["Retrieval (Table 3)", "Generation (Table 4)", "Learning (Tables 5–7)", "Runs"])

# --------------------------------------------------------------------------- #
with tab_r:
    path = tables / "retrieval_ablation.csv"
    if not path.exists():
        st.warning("Run `python -m src.evaluation.evaluate_retrieval "
                   "--config configs/ablation.yaml`")
    else:
        df = pd.read_csv(path)
        metric = st.selectbox("Metric", ["P@5", "MRR", "R@10", "P@1", "nDCG@5"],
                              index=0)
        plot_df = df[df.query_type != "ALL"]
        st.plotly_chart(px.bar(plot_df, x="query_type", y=metric, color="retriever",
                               barmode="group", title=f"{metric} by query language"),
                        use_container_width=True)
        st.dataframe(df, use_container_width=True)

        per_query = tables / "retrieval_per_query.csv"
        if per_query.exists():
            st.subheader("Error analysis")
            pq = pd.read_csv(per_query)
            retriever = st.selectbox("Retriever", sorted(pq.retriever.unique()))
            failures = pq[(pq.retriever == retriever) & (pq["P@5"] == 0)]
            st.write(f"{len(failures)} queries with no relevant chunk in the top 5")
            st.dataframe(failures[["query", "query_type", "concept", "detected_type",
                                   "top1_chunk"]], use_container_width=True, height=320)

# --------------------------------------------------------------------------- #
with tab_g:
    path = tables / "table4_generation.csv"
    if not path.exists():
        st.warning("Run `python -m src.evaluation.evaluate_generation`")
    else:
        summary = pd.read_csv(path)
        st.dataframe(summary, use_container_width=True)
        melted = summary[summary.query_type != "Overall"].melt(
            id_vars="query_type",
            value_vars=[c for c in ["rougeL_f1", "grounding_rate", "citation_rate",
                                    "register_match", "retrieval_hit"]
                        if c in summary.columns])
        st.plotly_chart(px.bar(melted, x="query_type", y="value", color="variable",
                               barmode="group", title="Generation quality"),
                        use_container_width=True)
        meta = tables / "generation_meta.json"
        if meta.exists():
            st.caption(read_json(meta)["note"])

# --------------------------------------------------------------------------- #
with tab_l:
    st.markdown("Data analysis")
    upload = st.file_uploader("scores.csv", type=["csv"])
    scores_path = cfg.path("paths.processed_dir") / "study_scores.csv"
    scores = pd.read_csv(upload) if upload is not None else (
        pd.read_csv(scores_path) if scores_path.exists() else None)

    if scores is None:
        st.warning("No scores available. Run `python -m src.data.simulate_study_data`.")
    else:
        if upload is None:
            st.info("Findings: ")
        results = analyse_scores(scores, cfg.get("evaluation.alpha", 0.05))
        gain, anc = results["gain_t_test"], results["ancova"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gain difference", f"{gain['mean_diff']:.1f} pp",
                  f"CI {gain['ci_low']:.1f}–{gain['ci_high']:.1f}")
        c2.metric("Cohen's d", f"{gain['cohens_d']:.2f}")
        c3.metric(f"t({gain['df']})", f"{gain['t']:.2f}", f"p = {gain['p_value']:.1e}")
        c4.metric("ANCOVA adjusted", f"{anc['adjusted_difference']:.1f}",
                  f"partial η² = {anc['partial_eta_squared']}")

        scores = scores.assign(gain=scores.post_test - scores.pre_test)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.violin(scores, x="group", y="gain", box=True, points="all",
                                  title="Gain distribution"), use_container_width=True)
        c2.plotly_chart(px.scatter(scores, x="pre_test", y="post_test", color="group",
                                   trendline="ols", title="ANCOVA: post vs pre"),
                        use_container_width=True)

        st.subheader("Assumption checks (Table 5)")
        st.dataframe(pd.DataFrame(results["assumption_checks"]), use_container_width=True)
        st.subheader("Power")
        st.json(results["power"])

        sat_path = cfg.path("paths.processed_dir") / "satisfaction_responses.csv"
        if sat_path.exists():
            st.subheader("Satisfaction (Table 7)")
            subscales, items, alpha_c = satisfaction_tables(pd.read_csv(sat_path))
            st.plotly_chart(px.bar(subscales, x="mean", y="subscale", orientation="h",
                                   error_x="sd", range_x=[1, 5],
                                   title=f"Sub-scale means (Cronbach's α = {alpha_c})"),
                            use_container_width=True)
            st.dataframe(items, use_container_width=True, height=320)

# --------------------------------------------------------------------------- #
with tab_runs:
    runs = load_runs(cfg.get("tracking.experiment_name", "algosathi"))
    if not runs:
        st.info("No tracked runs yet. Every script logs one automatically.")
    else:
        st.write(f"{len(runs)} tracked runs")
        st.dataframe(pd.DataFrame([{
            "run": r["run_name"], "started": r["started_at"],
            "status": r.get("status"), "metrics": len(r.get("metrics", {})),
        } for r in runs]), use_container_width=True)
        chosen = st.selectbox("Inspect run", [r["run_id"] for r in runs])
        st.json(next(r for r in runs if r["run_id"] == chosen)["metrics"])

    report = cfg.path("paths.results_dir") / "REPORT.md"
    if report.exists():
        st.subheader("Reproduction report")
        st.markdown(report.read_text(encoding="utf-8"))
