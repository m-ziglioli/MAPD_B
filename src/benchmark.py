"""
benchmark.py
============
Funzioni per eseguire ed esportare i risultati dei test di k-means||
distribuito su Dask.
"""

import os
import time

import dask
import dask.bag as db
import numpy as np
import pandas as pd

from src.kmeans_parallel import inertia_of_bag, kmeans_parallel

RESULTS_DIR = "/home/ubuntu/Project/libero_development/results"


def _build_bag(client, X, num_partitions):
    """Scatter dei dati ai worker e costruzione del Dask Bag dai futures."""
    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)
    return db.from_delayed([dask.delayed(f) for f in futures])


def calculate_inertia(X_bag, centroids):
    """Inertia calcolata con un task vettorizzato per partizione."""
    return inertia_of_bag(X_bag, centroids)


def run_single_test(client, k, l, r, num_partitions, max_iter_fit=10, seed=42, X=None, X_bag=None,
                     track_convergence=True, track_centroids=False):
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
            "partitions": num_partitions, "cost": None, "time": time.time() - start_time, "seed": seed,
            **({"cost_history": None, "iter_times": None} if track_convergence else {}),
            **({"n_centroids_history": None} if track_centroids else {})
        }, X_bag
    partial_time=time.time()
    clf.fit(X_bag, max_iter=max_iter_fit, track_convergence=track_convergence)
    end_time=time.time()
    elapsed_time = end_time - start_time
    lloyd_time=time.time()-partial_time

    cost = calculate_inertia(X_bag, clf.final_centroids)
    initial_cost = calculate_inertia(X_bag, clf.starting_centroids)
    
    print(f" -> Final cost: {cost:.2f} | Time: {elapsed_time:.2f}s | Time for LLoyd iterations only: {lloyd_time:.2f}s")

    result = {
        "k": k, "l": l, "r": r, "r_effective": getattr(clf, "n_rounds_", None),
        "partitions": num_partitions, "initial_cost": initial_cost, "final_cost": cost, "time": elapsed_time, "lloyd_time": lloyd_time, "num effective lloyd iters": getattr(clf, "n_iter_", None), "seed": seed
    }
    if track_convergence:
        result["cost_history"], result["iter_times"] = clf.cost_history_, clf.iter_times_
    if track_centroids:
        result["n_centroids_history"] = clf.n_centroids_history_
    # can return X_bag because lazy:
    return result, X_bag 


def run_benchmark(client, X_bag=None, combinations=None, k_values=None, label="benchmark",
                   max_iter_fit=10, seed=42, averaging_iterations=10, X_arr=None):
    """Esegue una griglia di test mantenendo i dati distribuiti su Dask.
    Varia n_workers, num_partitions, l_over_k, r.
    Salva ogni singolo risultato su CSV subito dopo il calcolo (append),
    cosi' che se l'esecuzione si interrompe nessun risultato gia' calcolato viene perso.
    """
    if X_bag is None and X_arr is None:
        raise ValueError("run_benchmark: fornire X_bag oppure X_arr")

    # Mantiene i dati distribuiti sui worker ed evita OOM/0-d array errors
    base_bag = build_bag(client, X_arr, combinations[0][1]) if X_bag is None else X_bag.map_partitions(lambda p: np.array(list(p))) # turn partitions into as many 2-d numpy as partitions arrays, for safety

    # Prepara il file CSV una sola volta, prima di iniziare i test
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{label}_{time.strftime('%Y%m%d%H%M%S')}.csv")
    csv_initialized = False

    results = []
    current_workers, current_partitions, current_bag = None, None, base_bag

    for k in k_values:
        for n_workers, num_partitions, l_over_k, r in combinations:
            if n_workers != current_workers:
                current_workers, current_partitions = n_workers, None
            if num_partitions != current_partitions:
                current_bag = build_bag(client, X_arr, num_partitions) if X_arr is not None else base_bag.repartition(npartitions=num_partitions)
                current_partitions = num_partitions

            l = max(1, round(l_over_k * k))
            print(f"Testing: k={k}, workers={n_workers}, partitions={current_bag.npartitions}, l={l} (l/k={l_over_k}), r={r}\n Iterating {averaging_iterations} times.")

            for i in range(averaging_iterations):
                print(f"Doing averaging iteration number {i}...")
                result, current_bag = run_single_test(
                    client, k=k, l=l, r=r, num_partitions=num_partitions,
                    max_iter_fit=max_iter_fit, seed=seed + i, X_bag=current_bag
                )
                result.update({"workers": n_workers, "l_over_k": l_over_k})
                results.append(result)

                # Salva subito questo singolo risultato (append su CSV)
                df_row = pd.DataFrame([result])
                df_row.to_csv(
                    csv_path,
                    mode="a",
                    header=not csv_initialized,
                    index=False
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
                "averaging_iterations": averaging_iterations
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