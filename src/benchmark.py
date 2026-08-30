"""
benchmark.py
============
Funzioni per eseguire ed esportare i risultati dei test di k-means||
distribuito su Dask.
"""

import os
import time

import dask
import numpy as np
import pandas as pd

from src.kmeans_parallel import inertia_of_bag, kmeans_parallel

RESULTS_DIR = "results"


def _build_bag(client, X, num_partitions):
    """Scatter dei dati ai worker e costruzione di una dask.array distribuita
    (un chunk 2D per partizione). I futures scatterati sono gia' matrici
    (m, d): nessun wrapper bag, nessun oggetto per riga."""
    from src.data_loader import _delayed_matrices_to_array

    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)
    delayed = [dask.delayed(f) for f in futures]
    return _delayed_matrices_to_array(delayed, [len(c) for c in chunks], X.shape[1])


def calculate_inertia(X, centroids):
    """Inertia calcolata con un task vettorizzato per partizione."""
    return inertia_of_bag(X, centroids)


def run_single_test(client, k, l, r, num_partitions, max_iter_fit=10, seed=42, X=None, X_bag=None,
                     track_convergence=False, track_centroids=False):
    """Esegue una singola run di k-means|| + Lloyd's fit."""
    if X_bag is None:
        if X is None:
            raise ValueError("Fornire 'X' oppure 'X_bag'.")
        X_bag = _build_bag(client, X, num_partitions)

    clf = kmeans_parallel(k=k, l=l, r=r)
    start_time = time.time()
    try:
        clf.compute_starting_centroids(X_bag, seed=seed, track_centroids=track_centroids)
    except ValueError as e:
        if "n_clusters" not in str(e):
            raise
        print(f" -> SEEDING FAILED (candidati < k): {e}")
        return {
            "k": k, "l": l, "r": r, "r_effective": getattr(clf, "n_rounds_", None),
            "partitions": num_partitions, "initial_cost": None, "final_cost": None,
            "time": time.time() - start_time, "lloyd_time": None, "seed": seed,
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
        "time": elapsed_time, "lloyd_time": lloyd_time, "seed": seed,
    }
    if track_convergence:
        result["cost_history"], result["iter_times"] = clf.cost_history_, clf.iter_times_
    if track_centroids:
        result["n_centroids_history"] = clf.n_centroids_history_
    return result, X_bag


def run_benchmark(client, X_bag=None, combinations=None, k_values=None, label="benchmark",
                   max_iter_fit=10, seed=42, averaging_iterations=10, X_arr=None):
    """Esegue una griglia di test e salva ogni singolo risultato su CSV subito
    dopo il calcolo (append), cosi' che un'interruzione non perda nulla.

    I dati vengono materializzati lato client UNA volta (se non gia' passati
    come ``X_arr``), poi ri-scatterati per ogni numero di partizioni."""
    if X_bag is None and X_arr is None:
        raise ValueError("run_benchmark: fornire X_bag oppure X_arr")

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
                    max_iter_fit=max_iter_fit, seed=seed + i, X_bag=current_bag
                )
                result.update({"workers": n_workers, "l_over_k": l_over_k})
                results.append(result)

                pd.DataFrame([result]).to_csv(
                    csv_path, mode="a", header=not csv_initialized, index=False
                )
                csv_initialized = True

    df_results = pd.DataFrame(results)
    print(f"\nRisultati completi salvati in: {csv_path}\n--- Benchmark Complete ---")

    for res in sorted(results, key=lambda x: (x.get("final_cost") is None, x.get("final_cost"))):
        print(res)

    return df_results


def combinations_fn(n_workers, l_over_k, r):
    # Each tuple: (n_workers, num_partitions, l_over_k, r)
    return [
        (n_workers, 8 * n_workers, l_over_k, r),
    ]


def run_worker_sweep(X_bag_or_arr, workers_list, combinations_fn, k_values,
                     label="worker_sweep", max_iter_fit=10, seed=42, averaging_iterations=10):
    """Sweep che varia workers e partizioni aprendo/chiudendo il cluster SSH."""
    from src.launch_cluster import launch_cluster, shutdown_cluster

    frames = []
    for n in workers_list:
        print(f"\n=== worker sweep: {n} workers ===")
        cluster, client = launch_cluster(n)
        try:
            kwargs = {
                "client": client, "combinations": combinations_fn(n), "k_values": k_values,
                "label": f"{label}_w{n}", "max_iter_fit": max_iter_fit, "seed": seed,
                "averaging_iterations": averaging_iterations,
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
