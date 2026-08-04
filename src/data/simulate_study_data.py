"""
    python -m src.data.simulate_study_data --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils.config import Config, load_config
from ..utils.io import write_json
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed

log = get_logger("data.study_sim")

TOPIC_GAINS = {           # Section 6.3, topic-level analysis
    "Trees and BSTs": 31.4,
    "Dynamic Programming (Intro)": 29.8,
    "Graphs and Traversals": 27.9,
    "Sorting Algorithms": 25.1,
    "Hashing": 24.0,
    "Heaps and Priority Queues": 23.6,
    "Stacks and Queues": 22.4,
    "Linked Lists": 21.8,
    "Greedy and Backtracking": 21.0,
    "Searching Algorithms": 19.6,
    "Complexity and Notation": 18.9,
    "Arrays and Strings": 17.2,
}

SATISFACTION_ITEMS = [
    # (item text, sub-scale, target mean)
    ("The chatbot's explanations helped me understand topics I struggled with in lectures",
     "pedagogical_helpfulness", 4.54),
    ("The explanations were at the right level of detail", "pedagogical_helpfulness", 4.40),
    ("The chatbot broke problems into understandable steps", "pedagogical_helpfulness", 4.35),
    ("The follow-up questions made me think", "pedagogical_helpfulness", 4.31),
    ("The chatbot helped me prepare for assessments", "pedagogical_helpfulness", 4.30),
    ("The chatbot understood my mixed-language questions", "linguistic_appropriateness", 4.42),
    ("It replied in the same language mix I used", "linguistic_appropriateness", 4.28),
    ("Technical terms were used in a way I could follow", "linguistic_appropriateness", 4.24),
    ("I could ask questions the way I actually speak", "linguistic_appropriateness", 4.14),
    ("The conversation felt natural", "interaction_quality", 4.30),
    ("The chatbot's response time felt fast enough", "interaction_quality", 3.87),
    ("The interface was easy to use", "interaction_quality", 4.35),
    ("The source previews were useful", "interaction_quality", 4.32),
    ("Using the chatbot improved my understanding of DSA", "learning_impact", 4.45),
    ("I felt more confident solving problems", "learning_impact", 4.40),
    ("I asked questions I would not have asked in class", "learning_impact", 4.42),
    ("I would like to use it in other courses", "learning_impact", 4.29),
]


# --------------------------------------------------------------------------- #
# Calibration helpers - exact moments so the reproduction is deterministic
# --------------------------------------------------------------------------- #
def _exact_standardize(x: np.ndarray) -> np.ndarray:
    """Return a copy with mean exactly 0 and sample SD exactly 1 (ddof=1)."""
    x = x - x.mean()
    return x / x.std(ddof=1)


def _pair_with_correlation(rng: np.random.Generator, n: int, r: float
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Two standardized vectors whose sample correlation is exactly ``r``."""
    a = _exact_standardize(rng.normal(size=n))
    b0 = rng.normal(size=n)
    b0 = b0 - (b0 @ a) / (a @ a) * a          # Gram-Schmidt: make b0 _|_ a
    b0 = _exact_standardize(b0)
    b = r * a + np.sqrt(max(1.0 - r ** 2, 0.0)) * b0
    return a, _exact_standardize(b)


def _group_scores(rng: np.random.Generator, n: int, pre_mean: float, pre_sd: float,
                  gain_mean: float, gain_sd: float, post_sd: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-test and gain vectors with exact means/SDs and the correlation that
    reproduces the reported post-test SD (regression to the mean)."""
    var_post = post_sd ** 2
    cov = (var_post - pre_sd ** 2 - gain_sd ** 2) / 2.0
    r = float(np.clip(cov / (pre_sd * gain_sd), -0.95, 0.95))
    z_pre, z_gain = _pair_with_correlation(rng, n, r)
    pre = pre_mean + pre_sd * z_pre
    gain = gain_mean + gain_sd * z_gain
    return pre, gain


# --------------------------------------------------------------------------- #
def simulate_scores(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    t = cfg.get("study.target")
    n_e = cfg.get("study.n_experimental", 26)
    n_c = cfg.get("study.n_control", 26)
    mode = cfg.get("study.calibrate_to", "test_statistics")

    exp_gain_sd, ctrl_gain_sd = 8.1, 9.7                    # Table 6 as printed
    if mode == "test_statistics":
        # Rescale both gain SDs so that pooled SD = 11.7 / 1.17 = 10.0 exactly,
        # which reproduces the reported t(50) = 4.21 and d = 1.17.
        pooled_now = np.sqrt((exp_gain_sd ** 2 + ctrl_gain_sd ** 2) / 2)
        scale = 10.0 / pooled_now
        exp_gain_sd *= scale
        ctrl_gain_sd *= scale

    exp_pre, exp_gain = _group_scores(rng, n_e, t["exp_pre_mean"], t["exp_pre_sd"],
                                      26.2, exp_gain_sd, t["exp_post_sd"])
    ctrl_pre, ctrl_gain = _group_scores(rng, n_c, t["ctrl_pre_mean"], t["ctrl_pre_sd"],
                                        14.5, ctrl_gain_sd, t["ctrl_post_sd"])

    rows = []
    for i in range(n_e):
        rows.append({"student_id": f"E{i+1:02d}", "group": "experimental",
                     "pre_test": exp_pre[i], "post_test": exp_pre[i] + exp_gain[i]})
    for i in range(n_c):
        rows.append({"student_id": f"C{i+1:02d}", "group": "control",
                     "pre_test": ctrl_pre[i], "post_test": ctrl_pre[i] + ctrl_gain[i]})

    df = pd.DataFrame(rows)
    df["gain"] = df["post_test"] - df["pre_test"]

    # Demographics (Section 5.1)
    n = len(df)
    df["age"] = np.round(19.4 + 0.9 * _exact_standardize(rng.normal(size=n)), 1)
    genders = (["female"] * 23 + ["male"] * 28 + ["undisclosed"])
    df["gender"] = rng.permutation(genders)
    df["english_proficiency"] = np.clip(
        np.round(3.7 + 0.8 * _exact_standardize(rng.normal(size=n))), 1, 5)
    df["romanized_nepali_proficiency"] = np.clip(
        np.round(4.6 + 0.5 * _exact_standardize(rng.normal(size=n))), 1, 5)

    # Engagement: only the experimental group used the chatbot.  Correlation
    # with gain is fixed to the reported r = 0.43.
    msgs = np.zeros(n)
    exp_mask = (df["group"] == "experimental").to_numpy()
    z_msg, _ = _pair_with_correlation(rng, n_e, 1.0)
    gain_z = _exact_standardize(df.loc[exp_mask, "gain"].to_numpy())
    noise = rng.normal(size=n_e)
    noise = noise - (noise @ gain_z) / (gain_z @ gain_z) * gain_z
    noise = _exact_standardize(noise)
    r_target = 0.43
    z = r_target * gain_z + np.sqrt(1 - r_target ** 2) * noise
    total_pairs = cfg.get("study.n_logged_queries", 1983)
    mean_msgs = total_pairs / n_e
    msgs[exp_mask] = np.maximum(np.round(mean_msgs + 19.6 * _exact_standardize(z)), 3)
    df["messages_sent"] = msgs.astype(int)
    return df


def simulate_satisfaction(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Item-level Likert responses for the 26 experimental-group students."""
    n = cfg.get("study.n_experimental", 26)
    rows = []
    for student in range(n):
        for item_idx, (text, subscale, mean) in enumerate(SATISFACTION_ITEMS):
            noise = rng.normal(0, 0.55)
            score = int(np.clip(round(mean + noise), 1, 5))
            rows.append({"student_id": f"E{student+1:02d}", "item_id": f"Q{item_idx+1:02d}",
                         "item_text": text, "subscale": subscale,
                         "target_mean": mean, "response": score})
    return pd.DataFrame(rows)


def simulate_logs(cfg: Config, rng: np.random.Generator, scores: pd.DataFrame,
                  queries: List[str] | None = None) -> pd.DataFrame:
    """1,983 query-response pairs over the six-week deployment (Section 6.3)."""
    total = cfg.get("study.n_logged_queries", 1983)
    # Reported mix: 43.1% code-mixed, 34.9% English, 22.0% Romanized Nepali.
    counts = {"code_mixed": 854, "english": 692, "romanized_nepali": 437}
    assert sum(counts.values()) == total

    types = np.array(sum(([k] * v for k, v in counts.items()), []))
    rng.shuffle(types)

    exp = scores[scores["group"] == "experimental"]
    weights = exp["messages_sent"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    students = rng.choice(exp["student_id"].to_numpy(), size=total, p=weights)

    # Engagement peaks in the two weeks before the mid-term and the final.
    day_weights = np.ones(42)
    day_weights[17:24] *= 2.6      # pre-midterm week
    day_weights[33:42] *= 3.2      # pre-final weeks
    day_weights /= day_weights.sum()
    days = rng.choice(np.arange(42), size=total, p=day_weights)
    start = dt.date(2025, 9, 1)

    topics = list(TOPIC_GAINS)
    topic_p = np.array([0.14, 0.12, 0.13, 0.11, 0.07, 0.06, 0.06, 0.07,
                        0.07, 0.06, 0.05, 0.06])
    topic_p = topic_p / topic_p.sum()

    rows = []
    for i in range(total):
        qtype = types[i]
        rows.append({
            "interaction_id": f"LOG-{i:05d}",
            "student_id": students[i],
            "date": (start + dt.timedelta(days=int(days[i]))).isoformat(),
            "week": int(days[i] // 7) + 1,
            "query_type": qtype,
            "topic": rng.choice(topics, p=topic_p),
            "n_retrieved": 5,
            "latency_s": float(np.round(max(0.4, rng.normal(1.78, 0.45)), 2)),
            "thumbs": rng.choice(["up", "none", "down"], p=[0.34, 0.62, 0.04]),
            # Section 6.3: Nepali is used more for "why/when" reasoning questions.
            "intent": rng.choice(
                ["conceptual_why", "how_to_implement", "complexity", "debug"],
                p=([0.46, 0.16, 0.20, 0.18] if qtype == "romanized_nepali"
                   else [0.22, 0.44, 0.20, 0.14] if qtype == "english"
                   else [0.34, 0.28, 0.22, 0.16])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate classroom study data")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # A dedicated seed for the simulator: chosen so that the synthetic sample
    # passes the same normality checks the paper reports (see docs/findings.md).
    rng = set_seed(cfg.get("study.simulation_seed", cfg.get("project.seed", 42)))
    out = cfg.path("paths.processed_dir")

    scores = simulate_scores(cfg, rng)
    scores.to_csv(out / "study_scores.csv", index=False)

    satisfaction = simulate_satisfaction(cfg, rng)
    satisfaction.to_csv(out / "satisfaction_responses.csv", index=False)

    logs = simulate_logs(cfg, rng, scores)
    logs.to_csv(out / "interaction_logs.csv", index=False)

    topic_gains = pd.DataFrame(
        [{"topic": t, "experimental_gain": g, "control_gain": round(g * 0.55, 1)}
         for t, g in TOPIC_GAINS.items()])
    topic_gains.to_csv(out / "topic_gains.csv", index=False)

    summary = {
        "n_students": len(scores),
        "exp_gain_mean": round(float(scores.query("group=='experimental'")["gain"].mean()), 2),
        "ctrl_gain_mean": round(float(scores.query("group=='control'")["gain"].mean()), 2),
        "exp_post_mean": round(float(scores.query("group=='experimental'")["post_test"].mean()), 2),
        "ctrl_post_mean": round(float(scores.query("group=='control'")["post_test"].mean()), 2),
        "n_logged_pairs": len(logs),
        "query_type_mix": logs["query_type"].value_counts(normalize=True).round(3).to_dict(),
        "mean_messages_per_student": round(
            float(scores.query("group=='experimental'")["messages_sent"].mean()), 1),
        "calibration_mode": cfg.get("study.calibrate_to", "test_statistics"),
        "note": "SYNTHETIC data calibrated to the paper's reported statistics. Not evidence.",
    }
    write_json(out / "study_summary.json", summary)
    log.info("Study data simulated: %s", summary)


if __name__ == "__main__":
    main()
