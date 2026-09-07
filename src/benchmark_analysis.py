"""
benchmark_analysis.py

Generic module to analyze benchmark results:
- group a CSV by arbitrary columns
- compute mean/std of arbitrary metrics
- generate errorbar plots for every combination of the "facet" columns
  against a variable on the x axis

Quick usage (an executable example is at the bottom of the file):

    from src.benchmark_analysis import BenchmarkAnalyzer

    analyzer = BenchmarkAnalyzer(
        data_path="results/kddcup99_benchmark_<timestamp>.csv",
        output_dir="figures",
        facet_cols=["k"],              # one figure per value (combination) of these columns
        x_col="l_over_k",               # variable on the x axis
        metrics=["cost", "time"],       # columns to compute mean/std of and plot
    )
    grouped = analyzer.compute_grouped_stats(groupby_cols=["k", "l_over_k"])
    analyzer.print_summary(grouped)
    analyzer.plot_all(grouped)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BenchmarkAnalyzer:
    data_path: str
    output_dir: str
    facet_cols: Sequence[str]          # columns defining a separate figure (e.g. ["k"])
    x_col: str                         # column on the x axis (e.g. "l_over_k")
    metrics: Sequence[str] = field(default_factory=lambda: ["cost", "time"])
    metric_labels: Optional[dict] = None   # e.g. {"cost": "Cost", "time": "Time (s)"}
    colors: Optional[dict] = None          # e.g. {"cost": "tab:blue", "time": "tab:orange"}
    dpi: int = 150
    df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self.df = pd.read_csv(self.data_path)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.metric_labels is None:
            self.metric_labels = {m: m.capitalize() for m in self.metrics}
        if self.colors is None:
            palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
            self.colors = {m: palette[i % len(palette)] for i, m in enumerate(self.metrics)}

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def compute_grouped_stats(self, groupby_cols: Sequence[str]) -> pd.DataFrame:
        """
        Group self.df by groupby_cols and compute mean/std/n_runs
        for each metric in self.metrics.
        """
        agg_dict = {}
        for m in self.metrics:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")
        # n_runs taken from the first available metric
        agg_dict["n_runs"] = (self.metrics[0], "size")

        grouped = (
            self.df.groupby(list(groupby_cols))
                   .agg(**agg_dict)
                   .reset_index()
                   .sort_values(list(groupby_cols))
        )
        self._last_grouped = grouped
        return grouped

    def print_summary(self, grouped: Optional[pd.DataFrame] = None) -> None:
        if grouped is None:
            grouped = self._last_grouped
        print(grouped.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def _facet_combinations(self, grouped: pd.DataFrame) -> Iterable[tuple]:
        """Generate all combinations present in the data for facet_cols."""
        if not self.facet_cols:
            yield tuple()
            return
        uniques = [sorted(grouped[c].unique()) for c in self.facet_cols]
        for combo in product(*uniques):
            yield combo

    def plot_all(self, grouped: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Generate one figure per combination of facet_cols, with one
        subplot per metric in self.metrics (errorbar mean ± std).
        Returns the list of saved paths.
        """
        if grouped is None:
            grouped = self._last_grouped

        saved_paths = []
        n_metrics = len(self.metrics)

        for combo in self._facet_combinations(grouped):
            if self.facet_cols:
                mask = pd.Series(True, index=grouped.index)
                for col, val in zip(self.facet_cols, combo):
                    mask &= grouped[col] == val
                sub = grouped[mask].sort_values(self.x_col)
                if sub.empty:
                    continue
                facet_str = "_".join(f"{c}_{v}" for c, v in zip(self.facet_cols, combo))
                title_str = ", ".join(f"{c}={v}" for c, v in zip(self.facet_cols, combo))
            else:
                sub = grouped.sort_values(self.x_col)
                facet_str = "all"
                title_str = ""

            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, self.metrics):
                ax.errorbar(
                    sub[self.x_col],
                    sub[f"{metric}_mean"],
                    yerr=sub[f"{metric}_std"],
                    marker="o", capsize=4, linestyle="-",
                    color=self.colors.get(metric, "tab:blue"),
                )
                ax.set_xlabel(self.x_col)
                ax.set_ylabel(f"{self.metric_labels.get(metric, metric)} (mean ± std)")
                ax.set_title(f"{self.metric_labels.get(metric, metric)} vs {self.x_col}"
                             + (f" ({title_str})" if title_str else ""))
                ax.grid(True, alpha=0.3)

            if title_str:
                fig.suptitle(title_str)
            fig.tight_layout()

            outpath = os.path.join(self.output_dir, f"{facet_str}_{self.x_col}.png")
            fig.savefig(outpath, dpi=self.dpi)
            plt.close(fig)
            saved_paths.append(outpath)
            print("Saved", outpath)

        return saved_paths
