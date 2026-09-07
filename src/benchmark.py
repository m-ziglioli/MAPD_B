"""
benchmark.py
============
Drivers to run and export benchmarks of distributed k-means|| on Dask.

Data distribution strategy
--------------------------
The dataset is materialized on the client ONCE (or passed directly as
``X_arr``) and re-scattered for each partitioning configuration: each chunk
is sent to a worker via ``client.scatter`` and referenced by a future, so
data never transits through the task graph as serialized task arguments.
The previous distribution is explicitly cancelled (``client.cancel``) when
the partition count changes, freeing worker memory.
"""

import os
import time

import dask
import numpy as np
import pandas as pd

from src.kmeans_parallel import inertia_of_bag, kmeans_parallel

RESULTS_DIR = "results"


def _build_bag(client, X, num_partitions):
    """Scatter the data to the workers and build a distributed dask.array
    (one 2-D chunk per partition). The scattered futures are already (m, d)
    matrices: no bag wrapper, no per-row objects."""
    from src.data_loader import _delayed_matrices_to_array

    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)
    delayed = [dask.delayed(f) for f in futures]
    return _delayed_matrices_to_array(delayed, [len(c) for c in chunks], X.shape[1])


def calculate_inertia(X, centroids):
    """Inertia computed with one vectorized task per partition."""
    return inertia_of_bag(X, centroids)


def run_single_test(client, k, l, r, num_partitions, max_iter_fit=10, seed=42, X=None, X_bag=None,
                     track_convergence=True, track_centroids=True, policy="auto"):
    """Run a single k-means|| seeding + Lloyd's fit.

    Returns (result_dict, X_bag): the distributed dataset is returned too,
    since it is lazy and can be reused by the caller without re-scattering.
    The only benign failure is the known "candidate pool < k" case (small
    l/k, low r), recorded with None costs; any other ValueError is a real
    bug and propagates.
    """
    if X_bag is None:
        if X is None:
            raise ValueError("Provide 'X' or 'X_bag'.")
        X_bag = _build_bag(client, X, num_partitions)

    clf = kmeans_parallel(k=k, l=l, r=r)
    start_time = time.time()
    try:
        clf.compute_starting_centroids(X_bag, seed=seed,
                                       track_centroids=track_centroids,
                                       policy=policy)
    except ValueError as e:
        if "n_clusters" not in str(e):
            raise
        print(f" -> SEEDING FAILED (candidates < k): {e}")
        return {
            "k": k, "l": l, "r": r, "r_effective": getattr(clf, "n_rounds_", None),
            "partitions": num_partitions, "initial_cost": None, "final_cost": None,
            "time": time.time() - start_time, "lloyd_time": None,
            "n_lloyd_iters": None, "seed": seed,
            **({"cost_history": None, "iter_times": None} if track_convergence else {}),
            **({"n_centroids_history": None} if track_centroids else {}),
        }, X_bag

    partial_time = time.time()
    clf.fit(X_bag, max_iter=max_iter_fit, track_convergence=track_convergence)
    end_time = time.time()
    elapsed_time = end_time - start_time
    lloyd_time = end_time - partial_time

    cost = calculate_inertia(X_bag, clf.final_centroids)
    initial_cost = calculate_inertia(X_bag, clf.starting_centroids)

    print(f" -> Final cost: {cost:.2f} | Time: {elapsed_time:.2f}s | "
          f"Time for Lloyd iterations only: {lloyd_time:.2f}s")

    result = {
        "k": k, "l": l, "r": r, "r_effective": getattr(clf, "n_rounds_", None),
        "partitions": num_partitions, "initial_cost": initial_cost, "final_cost": cost,
        "time": elapsed_time, "lloyd_time": lloyd_time,
        "n_lloyd_iters": getattr(clf, "n_iter_", None), "seed": seed,
    }
    if track_convergence:
        result["cost_history"], result["iter_times"] = clf.cost_history_, clf.iter_times_
    if track_centroids:
        result["n_centroids_history"] = clf.n_centroids_history_
    return result, X_bag


def run_benchmark(client, X_bag=None, combinations=None, k_values=None, label="benchmark",
                   max_iter_fit=10, seed=42, averaging_iterations=10, X_arr=None,
                   policy="auto"):
    """Run a grid of tests and append every single result to a CSV right
    after it is computed, so an interruption loses nothing already done.

    Repetition i uses seed + i (real averaging over seeds). If ``X_arr`` is
    not given, the dataset is materialized on the client once and
    re-scattered for each partition count.
    """
    if X_bag is None and X_arr is None:
        raise ValueError("run_benchmark: provide X_bag or X_arr")

    if X_arr is None:
        X_arr = X_bag.compute()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{label}{time.strftime('%Y%m%d%H%M%S')}.csv")
    csv_initialized = False

    results = []
    current_partitions = None
    current_bag = None

    for k in k_values:
        for n_workers, num_partitions, l_over_k, r in combinations:
            if num_partitions != current_partitions:
                old_bag = current_bag
                current_bag = _build_bag(client, X_arr, num_partitions)
                if old_bag is not None:
                    client.cancel(old_bag)
                current_partitions = num_partitions

            l = max(1, round(l_over_k * k))
            print(f"Testing: k={k}, workers={n_workers}, partitions={num_partitions}, "
                  f"l={l} (l/k={l_over_k}), r={r}\n Iterating {averaging_iterations} times.")

            for i in range(averaging_iterations):
                result, current_bag = run_single_test(
                    client, k=k, l=l, r=r, num_partitions=num_partitions,
                    max_iter_fit=max_iter_fit, seed=seed + i, X_bag=current_bag,
                    policy=policy
                )
                result.update({"workers": n_workers, "l_over_k": l_over_k})
                results.append(result)

                pd.DataFrame([result]).to_csv(
                    csv_path, mode="a", header=not csv_initialized, index=False
                )
                csv_initialized = True

    df_results = pd.DataFrame(results)
    print(f"\nFull results saved to: {csv_path}\n--- Benchmark Complete ---")

    for res in sorted(results, key=lambda x: (x.get("final_cost") is None, x.get("final_cost"))):
        print(res)

    return df_results


def combinations_fn(n_workers, l_over_k, r):
    # Each tuple: (n_workers, num_partitions, l_over_k, r)
    return [
        (n_workers, 8 * n_workers, l_over_k, r),
    ]


def run_worker_sweep(X_bag_or_arr, workers_list, combinations_fn, k_values,
                     label="worker_sweep", max_iter_fit=10, seed=42, averaging_iterations=10,
                     policy="auto"):
    """Sweep over worker counts, opening/closing the SSH cluster for each."""
    from src.launch_cluster import launch_cluster, shutdown_cluster

    frames = []
    for n in workers_list:
        print(f"\n=== worker sweep: {n} workers ===")
        cluster, client = launch_cluster(n)
        try:
            kwargs = {
                "client": client, "combinations": combinations_fn(n), "k_values": k_values,
                "label": f"{label}_w{n}", "max_iter_fit": max_iter_fit, "seed": seed,
                "averaging_iterations": averaging_iterations, "policy": policy,
            }
            if isinstance(X_bag_or_arr, np.ndarray):
                kwargs["X_arr"] = X_bag_or_arr
            else:
                kwargs["X_bag"] = X_bag_or_arr

            df_n = run_benchmark(**kwargs)
            df_n["workers_cfg"] = n
            frames.append(df_n)
        finally:
            shutdown_cluster(cluster, client)

    return pd.concat(frames, ignore_index=True)
