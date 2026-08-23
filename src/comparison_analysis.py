"""
comparison_analysis.py
========================
Tables and plots dedicated to the k-means|| vs k-means++ vs Random
comparison (results of kmeans_comparison.run_comparison). Separate from
benchmark_analysis.py, which is built around "continuous metric vs
numeric parameter" line plots for the partitioning sweeps -- here the
comparison is between categories (algorithm variants), a different shape
of problem that doesn't fit that module cleanly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _add_label_column(df):
    """Human-readable row label: method name, or "kmeans|| (l/k=.., r=..)"
    to tell apart different k-means|| configurations under comparison."""
    df = df.copy()

    def label(row):
        if row["method"] == "kmeans||":
            return f"kmeans|| (l/k={row['l_over_k']:g}, r={int(row['r'])})"
        return row["method"]

    df["label"] = df.apply(label, axis=1)
    return df


def summarize_comparison(df, value_cols=("cost_seed", "cost_final", "n_lloyd_iters")):
    """Group comparison results by (label, k) and compute mean/std/median
    for each value column -- long format, one row per algorithm
    variant/k combination."""
    df = _add_label_column(df)

    agg_dict = {}
    for col in value_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
        agg_dict[f"{col}_median"] = (col, "median")
    agg_dict["n_runs"] = (value_cols[0], "size")

    grouped = df.groupby(["label", "k"]).agg(**agg_dict).reset_index()
    return grouped.sort_values(["k", "label"]).reset_index(drop=True)


def format_paper_table(df, stat="median"):
    """Pivot into the paper's Table 1/2 layout: rows = algorithm variant,
    columns = one block per k with seed/final side by side.

    stat : "median" (paper convention) or "mean".
    """
    df = _add_label_column(df)

    pivoted = df.pivot_table(
        index="label", columns="k",
        values=["cost_seed", "cost_final"], aggfunc=stat,
    )
    pivoted = pivoted.reorder_levels([1, 0], axis=1)
    pivoted = pivoted.rename(columns={"cost_seed": "seed", "cost_final": "final"}, level=1)

    k_values = sorted(df["k"].unique())
    ordered_cols = pd.MultiIndex.from_product([k_values, ["seed", "final"]])
    pivoted = pivoted.reindex(columns=ordered_cols)
    return pivoted


def plot_cost_by_method(df, metric="cost_final", output_path=None, dpi=150):
    """Grouped bar chart: one group of bars per k, one bar per algorithm
    variant (mean +/- std) -- the right chart for comparing categories,
    unlike a continuous line plot."""
    df = _add_label_column(df)
    grouped = df.groupby(["label", "k"])[metric].agg(["mean", "std"]).reset_index()

    k_values = sorted(grouped["k"].unique())
    labels = sorted(grouped["label"].unique())

    x = np.arange(len(k_values))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(labels):
        sub = grouped[grouped["label"] == label].set_index("k").reindex(k_values)
        ax.bar(x + i * width, sub["mean"], width, yerr=sub["std"], capsize=3, label=label)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by method")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi)
        print("Saved", output_path)
    return fig


def plot_cost_vs_rounds(df, k, metric="cost_final", stat="median",
                        output_path=None, dpi=150):
    """Reproduce paper Figure 5.2/5.3: cost vs number of rounds r for
    k-means||, one line per l/k ratio, with horizontal reference lines for
    the serial k-means++/Random cost at this k. Only meaningful if the
    parallel_combinations used for run_comparison swept r at a fixed
    l/k ratio.

    stat : aggregation over repeated seeds. Default "median" follows the
    paper's protocol (Bahmani et al. report medians over 11 runs); use
    "mean" for the old behaviour.
    """
    sub = df[(df["k"] == k) & (df["method"] == "kmeans||")]
    if sub.empty:
        raise ValueError(f"No kmeans|| rows found for k={k}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for l_over_k, group in sub.groupby("l_over_k"):
        stats = group.groupby("r")[metric].agg(stat).sort_index()
        ax.plot(stats.index, stats.values, marker="o", label=f"l/k={l_over_k:g}")

    for method, style in (("k-means++", "--"), ("random", ":")):
        baseline_rows = df[(df["k"] == k) & (df["method"] == method)][metric]
        if not baseline_rows.empty:
            ax.axhline(baseline_rows.agg(stat), linestyle=style, color="black", label=method)

    ax.set_xlabel("number of rounds (r)")
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.set_title(f"Cost vs rounds (k={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi)
        print("Saved", output_path)
    return fig
