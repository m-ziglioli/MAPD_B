"""
kmeans_comparison.py
=====================
Driver to compare k-means|| against serial k-means++ and Random seeding,
following the evaluation setup of Bahmani et al. (VLDB 2012): cost is
reported both right after seeding ("seed") and after Lloyd's converges
("final"), averaged over repeated runs with different seeds.

Reuses kmeans_parallel/kmeans_serial for the algorithms and
calculate_inertia/_build_bag from benchmark.py for the cost formula and
data scattering, so the same cost is computed identically for all three
methods -- no existing file is modified.

Wall-clock time is recorded for reference only: kmeans_serial runs
single-threaded on the client, kmeans_parallel runs distributed on the
Dask cluster, so times are not a controlled, apples-to-apples comparison.
"""

import os
import time

import dask
import numpy as np
import pandas as pd

from benchmark import RESULTS_DIR, _build_bag, calculate_inertia
from kmeans_parallel import kmeans_parallel
from kmeans_serial import kmeans_serial


def _materialize_bag(client, X_bag):
    """Gather a Dask Bag of per-row arrays into a single numpy array on
    the client. Each partition is vstack-ed on the worker first, so only
    a handful of compact blocks travel over the network instead of one
    tiny object per row (see CHANGES.md for the CommClosedError this
    avoids)."""
    delayed_partitions = [dask.delayed(np.vstack)(p) for p in X_bag.to_delayed()]
    partition_futures = client.compute(delayed_partitions)
    partitions = client.gather(partition_futures)
    return np.vstack(partitions)


def pilot_timing_check(X_arr, k, init="k-means++", seed=42, n_local_trials=None,
                        max_iter=100, tol=1e-4):
    """Time a single serial run (seed + fit) at a given k. Meant to be run
    once at the smallest planned k before committing to the full
    comparison loop, since greedy k-means++ seeding can be slow at this
    dataset's scale."""
    clf = kmeans_serial(k=k, init=init)
    t0 = time.time()
    clf.compute_starting_centroids(X_arr, seed=seed, n_local_trials=n_local_trials)
    t_seed = time.time() - t0
    clf.fit(X_arr, max_iter=max_iter, tol=tol)
    t_fit = time.time() - t0 - t_seed
    print(f"[{init}] k={k}: seed={t_seed:.1f}s, fit={t_fit:.1f}s, total={t_seed + t_fit:.1f}s")
    return t_seed, t_fit


def run_comparison(client, X_bag, k_values, parallel_combinations, seed=42,
                    averaging_iterations=11, max_iter_fit=100, tol=1e-4,
                    n_local_trials=None, label="kmeans_comparison"):
    """
    Compare k-means||, serial k-means++ and Random over k_values.

    Parameters
    ----------
    parallel_combinations : list of (num_partitions, l_over_k, r)
        k-means|| configurations to test, chosen by the caller.
    averaging_iterations : int
        Number of repeated runs per (k, method) combination; the paper
        uses 11 for cost tables (median).
    max_iter_fit, tol : Lloyd's stopping criteria, shared by all three
        methods so "final cost" means "cost at convergence" everywhere.
    n_local_trials : passed to the serial k-means++ seeding (see
        kmeans_serial.compute_starting_centroids).

    Returns
    -------
    df_results : pd.DataFrame, also saved to results/{label}_<timestamp>.csv
    """
    X_arr = _materialize_bag(client, X_bag)

    rows = []
    current_partitions = None
    current_bag = None

    for k in k_values:
        for num_partitions, l_over_k, r in parallel_combinations:
            l = max(1, round(l_over_k * k))

            if num_partitions != current_partitions:
                old_bag = current_bag
                current_bag = _build_bag(client, X_arr, num_partitions)
                if old_bag is not None:
                    client.cancel(old_bag)
                current_partitions = num_partitions

            print(f"[kmeans||] k={k}, partitions={num_partitions}, l={l} (l/k={l_over_k}), r={r}")
            for i in range(averaging_iterations):
                run_seed = seed + i
                clf = kmeans_parallel(k=k, l=l, r=r)
                t0 = time.time()
                clf.compute_starting_centroids(current_bag, seed=run_seed)
                t_seed = time.time() - t0
                clf.fit(current_bag, max_iter=max_iter_fit, tol=tol, track_convergence=True)
                t_fit = time.time() - t0 - t_seed

                rows.append({
                    "method": "kmeans||",
                    "k": k, "l": l, "r": r,
                    "partitions": num_partitions, "l_over_k": l_over_k,
                    "seed": run_seed,
                    "cost_seed": calculate_inertia(current_bag, clf.starting_centroids),
                    "cost_final": calculate_inertia(current_bag, clf.final_centroids),
                    "n_lloyd_iters": len(clf.iter_times_),
                    "time_seed": t_seed, "time_fit": t_fit,
                })

        print(f"[serial] k={k}: k-means++ and random, {averaging_iterations} repetitions each")
        for init in ("k-means++", "random"):
            for i in range(averaging_iterations):
                run_seed = seed + i
                clf = kmeans_serial(k=k, init=init)
                t0 = time.time()
                clf.compute_starting_centroids(X_arr, seed=run_seed, n_local_trials=n_local_trials)
                t_seed = time.time() - t0
                clf.fit(X_arr, max_iter=max_iter_fit, tol=tol)
                t_fit = time.time() - t0 - t_seed

                rows.append({
                    "method": init,
                    "k": k, "l": np.nan, "r": np.nan,
                    "partitions": np.nan, "l_over_k": np.nan,
                    "seed": run_seed,
                    "cost_seed": calculate_inertia(current_bag, clf.starting_centroids),
                    "cost_final": calculate_inertia(current_bag, clf.final_centroids),
                    "n_lloyd_iters": clf.n_iter_,
                    "time_seed": t_seed, "time_fit": t_fit,
                })

    df_results = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"{label}_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df_results
