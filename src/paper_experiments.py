"""
paper_experiments.py
====================
Drivers for the reproduction of Bahmani et al. (2012), "Scalable
K-Means++" (docs/1203.6402v1.pdf), following the plan frozen in
docs/ANALYSIS_PLAN.md.

Artifacts covered (Partition baseline excluded everywhere):
- run_fig51  : Fig 5.1 — final cost vs number of rounds r, KDD 10%,
               EXACT sampling of l points per round (sampling="exact"),
               k in {17,33,65,129}, l/k in {1,2,4};
- run_fig52  : Fig 5.2 — final cost vs number of rounds r (from 0) on the
               synthetic GaussMixture, k=50, horizontal k-means++
               reference; r=0 degrades to k uniform centers (random-
               baseline path);
- run_table34: Tables 3 and 4 — KDD full, k in {500,1000}, r=5,
               l/k in {0.1,...,10} + Random baseline; seed/final cost
               (raw, scaled x1e-10 at analysis time) and times (Table 4).

Experimental protocol (from the paper and ANALYSIS_PLAN.md):
- median over n_runs repetitions (default 11), seed of run i = seed + i;
- "final" costs after Lloyd's convergence (shared max_iter_fit/tol);
- ATTENTION to policies: Tables 3/4 use policy="fixed", r=5 — the automatic
  rule "l/k<=0.1 -> 15 rounds" must NOT apply there (the paper uses r=5
  also for l/k=0.1); Fig 5.2 sweeps explicitly over r=0..15 so it always
  uses policy="fixed"; Fig 5.1 has l/k>=1 so r is never rewritten by the
  automatic rule.
- The known "candidate pool < k" case (can happen with small l/k and low
  r: expected pool ~ 1 + r*l) does NOT interrupt the sweeps: the run is
  recorded with NaN costs and failed=True (robustness for overnight runs),
  as done in benchmark.run_single_test.

CSVs go to results/ (gitignored), figures to figures/.
All functions accept client=None: they run on the local scheduler (useful
for validations on reduced grids before the cluster sessions).
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.benchmark import RESULTS_DIR
from src.data_loader import array_to_dask, make_gauss_mixture
from src.kmeans_parallel import kmeans_parallel, inertia_of_bag
from src.kmeans_serial import kmeans_serial


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_results(df, label):
    """Save the results DataFrame to results/{label}_{timestamp}.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{label}_{timestamp}.csv")
    df.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")
    return path


def _load_csv(path):
    return pd.read_csv(path)


def _inertia_numpy(X, C):
    """Client-side inertia (small arrays: GaussMixture 10k x 15)."""
    C = np.asarray(C, dtype=np.float64)
    m_sq = np.einsum("ij,ij->i", X, X)
    c_sq = np.einsum("ij,ij->i", C, C)
    d2 = m_sq[:, None] + c_sq[None, :] - 2.0 * (X @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return float(d2[np.arange(X.shape[0]), d2.argmin(axis=1)].sum())


def _run_one_parallel(X_bag, k, l, r, run_seed, policy="auto",
                      sampling="bernoulli", max_iter_fit=100, tol=1e-4):
    """Single k-means|| run (+ Lloyd's) with benign handling of the known
    'candidate pool < k' case: sklearn raises ValueError in the
    reclustering and the sweep continues recording the row with failed=True
    and NaN costs. Any OTHER ValueError is a real error and propagates.

    Returns a dict with: r_effective, cost_seed, cost_final,
    n_lloyd_iters, time_seed, time_fit, failed.
    """
    clf = kmeans_parallel(k=k, l=l if l is not None else 1, r=r)
    out = {"r_effective": None, "cost_seed": np.nan, "cost_final": np.nan,
           "n_lloyd_iters": np.nan, "time_seed": np.nan, "time_fit": np.nan,
           "failed": True}
    t0 = time.time()
    try:
        clf.compute_starting_centroids(X_bag, seed=run_seed,
                                       policy=policy, sampling=sampling)
    except ValueError as e:
        # Same criterion as benchmark.run_single_test: only the known
        # error ("n_samples=X should be >= n_clusters=Y") is benign
        if "n_clusters" not in str(e):
            raise
        print(f" -> SEEDING FAILED (candidates < k, k={k}, l={l}, r={r}): {e}")
        return out
    out["failed"] = False
    out["r_effective"] = clf.n_rounds_
    out["time_seed"] = time.time() - t0
    clf.fit(X_bag, max_iter=max_iter_fit, tol=tol, track_convergence=True)
    out["time_fit"] = time.time() - t0 - out["time_seed"]
    out["cost_seed"] = inertia_of_bag(X_bag, clf.starting_centroids)
    out["cost_final"] = inertia_of_bag(X_bag, clf.final_centroids)
    out["n_lloyd_iters"] = len(clf.iter_times_)
    return out


# ---------------------------------------------------------------------------
# Fig 5.1 — KDD 10%, exact-l sampling
# ---------------------------------------------------------------------------

def run_fig51(client, X_bag, k_values=(17, 33, 65, 129),
              l_over_k_values=(1, 2, 4), r_values=tuple(range(1, 11)),
              n_runs=11, seed=42, max_iter_fit=100, tol=1e-4,
              num_partitions=None, label="fig51"):
    """Final cost vs number of rounds r, sampling EXACTLY l points per
    round (sampling="exact", Fig 5.1 protocol). Median over n_runs at
    analysis/plot time; here one row per single run.

    num_partitions is only informative (a label in the results): the
    distributed dataset arrives already built by the caller.
    """
    rows = []
    for k in k_values:
        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for r in r_values:
                # l/k >= 1: the automatic rule never touches r; the policy
                # is fixed to make explicit that the protocol requires
                # THIS r
                for i in range(n_runs):
                    res = _run_one_parallel(X_bag, k, l, r, seed + i,
                                            policy="fixed", sampling="exact",
                                            max_iter_fit=max_iter_fit, tol=tol)
                    rows.append({
                        "artifact": "fig51",
                        "method": "kmeans||",
                        "k": k, "l": l, "r": r, "l_over_k": l_over_k,
                        "partitions": num_partitions,
                        "sampling": "exact",
                        "run": i, "seed": seed + i,
                        **res,
                    })
                print(f"[fig51] k={k}, l={l} (l/k={l_over_k}), r={r}: "
                      f"{n_runs} runs completed")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 5.2 — GaussMixture, r from 0, k-means++ as reference
# ---------------------------------------------------------------------------

def run_fig52(client=None, R_values=(1, 10, 100),
              l_over_k_values=(0.1, 0.5, 1, 2, 10), r_values=tuple(range(0, 16)),
              k=50, n=10_000, d=15, n_runs=11, seed=42,
              max_iter_fit=100, tol=1e-4, n_partitions=8,
              include_kmpp_reference=True, label="fig52"):
    """Final cost vs r on GaussMixture (Fig 5.2). No cluster needed: with
    n=10^4 points it runs comfortably also locally (client=None).

    r=0 -> k uniform centers (policy="fixed", random-baseline path): the
    axis starts at the level of the Random baseline, as in the paper.
    The dataset is regenerated with a deterministic seed for each R
    (gm_seed = seed + R).
    """
    rows = []
    for R in R_values:
        gm_seed = seed + int(R)
        X, _, _ = make_gauss_mixture(n=n, k=k, d=d, R=R, seed=gm_seed)
        X_bag = array_to_dask(X, n_partitions)

        if include_kmpp_reference:
            print(f"[fig52] R={R}: k-means++ reference, {n_runs} runs")
            for i in range(n_runs):
                run_seed = seed + i
                srl = kmeans_serial(k=k, init="k-means++")
                srl.compute_starting_centroids(X, seed=run_seed)
                srl.fit(X, max_iter=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "fig52",
                    "method": "k-means++",
                    "R": R, "n": n, "d": d, "k": k,
                    "l": np.nan, "r": np.nan, "l_over_k": np.nan,
                    "sampling": "bernoulli",
                    "run": i, "seed": run_seed,
                    "r_effective": np.nan,
                    "cost_seed": np.nan,
                    "cost_final": _inertia_numpy(X, srl.final_centroids),
                    "n_lloyd_iters": srl.n_iter_,
                    "time_seed": np.nan, "time_fit": np.nan,
                    "failed": False,
                })

        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for r in r_values:
                for i in range(n_runs):
                    res = _run_one_parallel(X_bag, k, l, r, seed + i,
                                            policy="fixed",
                                            max_iter_fit=max_iter_fit, tol=tol)
                    rows.append({
                        "artifact": "fig52",
                        "method": "kmeans||",
                        "R": R, "n": n, "d": d, "k": k,
                        "l": l, "r": r, "l_over_k": l_over_k,
                        "sampling": "bernoulli",
                        "run": i, "seed": seed + i,
                        **res,
                    })
            print(f"[fig52] R={R}, l={l} (l/k={l_over_k}): r sweep completed")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 3 and 4 — KDD full
# ---------------------------------------------------------------------------

def run_table34(client, X_bag_full, k_values=(500, 1000),
                l_over_k_values=(0.1, 0.5, 1, 2, 10), r_fixed=5,
                n_runs=11, seed=42, max_iter_fit=100, tol=1e-4,
                num_partitions=64, include_random=True, label="table34"):
    """Tables 3 (cost) and 4 (times) on KDD full.

    Critical protocol: policy="fixed", r=r_fixed (=5 as in the paper).
    With policy="auto" the l/k=0.1 configuration would be forced to 15
    rounds, which is NOT the protocol of the table.

    include_random=True adds the Random baseline as an r=0 path (k uniform
    centers, no reclustering): same Lloyd's budget. Records cost_seed AND
    cost_final (the table reports both, raw; the x1e-10 scale is applied at
    analysis time) plus the separate seeding/Lloyd's times for Table 4.
    """
    rows = []
    for k in k_values:
        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for i in range(n_runs):
                res = _run_one_parallel(X_bag_full, k, l, r_fixed, seed + i,
                                        policy="fixed", sampling="bernoulli",
                                        max_iter_fit=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "table34",
                    "method": "kmeans||",
                    "k": k, "l": l, "r": r_fixed, "l_over_k": l_over_k,
                    "partitions": num_partitions, "sampling": "bernoulli",
                    "run": i, "seed": seed + i,
                    **res,
                })
            print(f"[table34] k={k}, l={l} (l/k={l_over_k}), r={r_fixed}: "
                  f"{n_runs} runs completed")
        if include_random:
            for i in range(n_runs):
                res = _run_one_parallel(X_bag_full, k, None, 0, seed + i,
                                        policy="fixed", sampling="bernoulli",
                                        max_iter_fit=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "table34",
                    "method": "random",
                    "k": k, "l": np.nan, "r": 0, "l_over_k": np.nan,
                    "partitions": num_partitions, "sampling": "bernoulli",
                    "run": i, "seed": seed + i,
                    **res,
                })
            print(f"[table34] k={k}: Random baseline, {n_runs} runs completed")

    df = pd.DataFrame(rows)
    df["cost_seed_e10"] = df["cost_seed"] / 1e10
    df["cost_final_e10"] = df["cost_final"] / 1e10
    return df


# ---------------------------------------------------------------------------
# Plotting (log-y, medians: paper conventions)
# ---------------------------------------------------------------------------

def _agg_median(df, by, metric="cost_final"):
    return df.groupby(by)[metric].median().sort_index()


def plot_fig51(results, output_dir="figures", dpi=150):
    """Fig 5.1: grid of panels (one per k), median curves per l/k,
    logarithmic y axis. `results` is a DataFrame or the path of the CSV."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    sub = df[df["artifact"] == "fig51"]
    k_values = sorted(sub["k"].unique())
    l_over_k_values = sorted(sub["l_over_k"].unique())

    ncols = 2
    nrows = int(np.ceil(len(k_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows),
                             squeeze=False)
    for ax, k in zip(axes.ravel(), k_values):
        for l_over_k in l_over_k_values:
            sel = sub[(sub["k"] == k) & (sub["l_over_k"] == l_over_k)]
            if sel.empty:
                continue
            med = _agg_median(sel, "r")
            ax.plot(med.index, med.values, marker="o", label=f"ℓ/k={l_over_k:g}")
        ax.set_yscale("log")
        ax.set_xlabel("number of rounds $r$")
        ax.set_ylabel("final cost (median)")
        ax.set_title(f"$k={k}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(k_values):]:
        ax.axis("off")
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig51_cost_vs_rounds.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print("Saved", outpath)
    return outpath


def plot_fig52(results, output_dir="figures", dpi=150):
    """Fig 5.2: one panel per R, median curves per l/k, horizontal
    k-means++ line (median of the reference runs), x axis from r=0,
    logarithmic y."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    sub = df[df["artifact"] == "fig52"]
    par = sub[sub["method"] == "kmeans||"]
    R_values = sorted(par["R"].unique())
    l_over_k_values = sorted(par["l_over_k"].dropna().unique())

    ncols = len(R_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.6), squeeze=False)
    for ax, R in zip(axes.ravel(), R_values):
        for l_over_k in l_over_k_values:
            sel = par[(par["R"] == R) & (par["l_over_k"] == l_over_k)]
            if sel.empty:
                continue
            med = _agg_median(sel, "r")
            ax.plot(med.index, med.values, marker="o", label=f"ℓ/k={l_over_k:g}")
        ref = sub[(sub["R"] == R) & (sub["method"] == "k-means++")]["cost_final"]
        if not ref.empty:
            ax.axhline(ref.median(), linestyle="--", color="black",
                       label="k-means++")
        ax.set_yscale("log")
        ax.set_xlabel("number of rounds $r$")
        ax.set_ylabel("final cost (median)")
        ax.set_title(f"$R={R:g}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig52_cost_vs_rounds.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print("Saved", outpath)
    return outpath


def table34_cost_table(results, stat="median"):
    """Table 3 layout: rows = algorithm variant, columns = one block per k
    with seed/final side by side, costs scaled x1e-10 (as in the paper).
    Reuses the pivot of comparison_analysis.format_paper_table."""
    from src.comparison_analysis import format_paper_table

    df = _load_csv(results) if isinstance(results, str) else results.copy()
    piv = format_paper_table(df, stat=stat)
    return piv / 1e10


def table34_time_table(results, stat="median"):
    """Table 4 layout: average times (init + Lloyd) per method and k."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    df = df.copy()
    df["label"] = df.apply(
        lambda r: r["method"] if r["method"] != "kmeans||"
        else f"kmeans|| (l/k={r['l_over_k']:g}, r={int(r['r'])})", axis=1)
    grouped = df.groupby(["label", "k"])[["time_seed", "time_fit"]].agg(stat)
    grouped["time_total"] = grouped["time_seed"] + grouped["time_fit"]
    return grouped
