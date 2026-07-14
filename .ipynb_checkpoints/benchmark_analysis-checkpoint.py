"""
benchmark_analysis.py

Modulo generico per analizzare risultati di benchmark:
- raggruppa un CSV per colonne arbitrarie
- calcola mean/std di metriche arbitrarie
- genera grafici errorbar per ogni combinazione delle colonne "facet"
  rispetto a una variabile sull'asse x

Uso rapido (in fondo al file trovi un esempio eseguibile):

    from benchmark_analysis import BenchmarkAnalyzer

    analyzer = BenchmarkAnalyzer(
        data_path="/home/ubuntu/Project/working/results/kddcup99_benchmark_20260714_040919.csv",
        output_dir="/home/ubuntu/Project/working/results/plots",
        facet_cols=["k"],              # una figura per ogni valore (combinazione) di queste colonne
        x_col="l_over_k",               # variabile sull'asse x
        metrics=["cost", "time"],       # colonne di cui calcolare mean/std e plottare
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
    facet_cols: Sequence[str]          # colonne che definiscono una figura separata (es. ["k"])
    x_col: str                         # colonna sull'asse x (es. "l_over_k")
    metrics: Sequence[str] = field(default_factory=lambda: ["cost", "time"])
    metric_labels: Optional[dict] = None   # es. {"cost": "Cost", "time": "Time (s)"}
    colors: Optional[dict] = None          # es. {"cost": "tab:blue", "time": "tab:orange"}
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
    # Statistiche
    # ------------------------------------------------------------------ #
    def compute_grouped_stats(self, groupby_cols: Sequence[str]) -> pd.DataFrame:
        """
        Raggruppa self.df per groupby_cols e calcola mean/std/n_runs
        per ciascuna metrica in self.metrics.
        """
        agg_dict = {}
        for m in self.metrics:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")
        # n_runs preso dalla prima metrica disponibile
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
        """Genera tutte le combinazioni presenti nei dati per facet_cols."""
        if not self.facet_cols:
            yield tuple()
            return
        uniques = [sorted(grouped[c].unique()) for c in self.facet_cols]
        for combo in product(*uniques):
            yield combo

    def plot_all(self, grouped: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Genera una figura per ogni combinazione di facet_cols, con un
        subplot per ogni metrica in self.metrics (errorbar mean ± std).
        Ritorna la lista dei path salvati.
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
