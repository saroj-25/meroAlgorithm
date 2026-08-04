

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")          # headless-safe: scripts must run without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = {"experimental": "#2b6cb0", "control": "#a0aec0",
           "english": "#2b6cb0", "romanized_nepali": "#dd6b20",
           "code_mixed": "#38a169"}


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def plot_training_curves(history: Sequence[dict], path: str | Path) -> Path:
    """Loss and accuracy/F1 curves for the language-ID classifier."""
    df = pd.DataFrame(list(history))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train loss", marker="o", ms=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="val loss", marker="s", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy + L2")
    axes[0].set_title("Language-ID training loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="train acc", marker="o", ms=3)
    axes[1].plot(df["epoch"], df["val_acc"], label="val acc", marker="s", ms=3)
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="val macro-F1", marker="^", ms=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("Language-ID accuracy / F1"); axes[1].legend(); axes[1].grid(alpha=.3)
    return _save(fig, path)


def plot_retrieval_ablation(df: pd.DataFrame, path: str | Path,
                            metric: str = "P@5") -> Path:
    """Grouped bars: retriever configuration x query language (Table 3)."""
    pivot = df.pivot_table(index="query_type", columns="retriever", values=metric)
    labels = list(pivot.index)
    retrievers = list(pivot.columns)
    x = np.arange(len(labels))
    width = 0.8 / max(len(retrievers), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, retriever in enumerate(retrievers):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[retriever].to_numpy(),
               width, label=retriever)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(metric); ax.set_ylim(0, 1.02)
    ax.set_title(f"Retrieval {metric} by query language and retriever")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    return _save(fig, path)


def plot_learning_outcomes(scores: pd.DataFrame, path: str | Path) -> Path:
    """Pre/post means with SD error bars, plus the gain distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    groups = ["experimental", "control"]
    width = 0.35
    x = np.arange(2)
    for i, phase in enumerate(["pre_test", "post_test"]):
        means = [scores.loc[scores.group == g, phase].mean() for g in groups]
        sds = [scores.loc[scores.group == g, phase].std(ddof=1) for g in groups]
        axes[0].bar(x + i * width - width / 2, means, width, yerr=sds, capsize=4,
                    label=phase.replace("_", "-"))
    axes[0].set_xticks(x); axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("score (%)"); axes[0].set_title("Pre-test vs post-test")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.3)

    data = [scores.loc[scores.group == g, "gain"].to_numpy() for g in groups]
    parts = axes[1].violinplot(data, showmeans=True)
    for body, g in zip(parts["bodies"], groups):
        body.set_facecolor(PALETTE[g]); body.set_alpha(.6)
    axes[1].set_xticks([1, 2]); axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("gain (pp)"); axes[1].set_title("Learning-gain distribution")
    axes[1].grid(axis="y", alpha=.3)
    return _save(fig, path)


def plot_query_mix(logs: pd.DataFrame, path: str | Path) -> Path:
    """Query-language mix and weekly engagement (Sections 6.3 and 6.6)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    mix = logs["query_type"].value_counts()
    axes[0].pie(mix.to_numpy(), labels=list(mix.index), autopct="%1.1f%%",
                colors=[PALETTE.get(t, "#888") for t in mix.index], startangle=120)
    axes[0].set_title("Logged query language mix")

    weekly = logs.groupby(["week", "query_type"]).size().unstack(fill_value=0)
    weekly.plot(kind="bar", stacked=True, ax=axes[1],
                color=[PALETTE.get(c, "#888") for c in weekly.columns])
    axes[1].set_xlabel("week of deployment"); axes[1].set_ylabel("queries")
    axes[1].set_title("Engagement over time"); axes[1].legend(fontsize=8)
    return _save(fig, path)


def plot_satisfaction(subscales: pd.DataFrame, path: str | Path) -> Path:
    """Horizontal bars of satisfaction sub-scale means (Table 7)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    order = subscales.sort_values("mean")
    ax.barh(order["subscale"].str.replace("_", " "), order["mean"],
            xerr=order["sd"], color="#2b6cb0", alpha=.85, capsize=4)
    ax.set_xlim(1, 5); ax.axvline(4.31, ls="--", c="grey", lw=1,
                                  label="overall mean 4.31")
    ax.set_xlabel("mean rating (1-5)"); ax.set_title("Student satisfaction sub-scales")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)
    return _save(fig, path)


def plot_topic_gains(topic_gains: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    df = topic_gains.sort_values("experimental_gain")
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["experimental_gain"], 0.4, label="experimental",
            color=PALETTE["experimental"])
    ax.barh(y + 0.2, df["control_gain"], 0.4, label="control", color=PALETTE["control"])
    ax.set_yticks(y); ax.set_yticklabels(df["topic"], fontsize=8)
    ax.set_xlabel("mean gain (pp)"); ax.set_title("Learning gain by topic")
    ax.legend(); ax.grid(axis="x", alpha=.3)
    return _save(fig, path)


def plot_hparam_sweep(df: pd.DataFrame, path: str | Path, x: str,
                      metric: str = "P@5") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for qtype, sub in df.groupby("query_type"):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[metric], marker="o", label=qtype,
                color=PALETTE.get(qtype, None))
    ax.set_xlabel(x); ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {x}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_generation_quality(df: pd.DataFrame, path: str | Path) -> Path:
    metrics = [c for c in ["bertscore_f1", "rougeL_f1", "grounding_rate",
                           "register_match", "citation_rate"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(df))
    width = 0.8 / max(len(metrics), 1)
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width - 0.4 + width / 2, df[metric].to_numpy(), width, label=metric)
    ax.set_xticks(x); ax.set_xticklabels(df["query_type"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Generation quality by query language"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    return _save(fig, path)
