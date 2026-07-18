"""
benchmark.py
============
Funzioni per eseguire ed esportare i risultati dei test di k-means||
distribuito su Dask.

Due modalità d'uso, entrambe pensate per essere chiamate dal notebook:

1. run_single_test: esegue UNA singola combinazione (k, l, r, workers,
   partitions) e ritorna costo/tempo.
2. run_benchmark: esegue una GRIGLIA di combinazioni (come i vecchi
   main/main_2 del notebook originale) e salva i risultati in CSV
   dentro ./results.
"""

import os
import time

import dask
import dask.bag as db
import numpy as np
import pandas as pd

from kmeans_parallel import kmeans_parallel

RESULTS_DIR = "results"


def _build_bag(client, X, num_partitions):
    """Scatter dei dati ai worker e costruzione del Dask Bag dai futures
    (evita di far passare l'intero array attraverso il task graph)."""
    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)
    return db.from_delayed([dask.delayed(f) for f in futures])


def calculate_inertia(X_bag, centroids):
    centroids_arr = np.vstack(centroids)
    return X_bag.map(lambda x: np.min(np.linalg.norm(x - centroids_arr, axis=1) ** 2)).sum().compute()


def run_single_test(client, k, l, r, num_partitions, max_iter_fit=10, seed=42, X=None, X_bag=None):
    """
    Esegue una singola run di k-means|| + Lloyd's fit su una combinazione
    di parametri, e ritorna un dict con i risultati.

    Se X_bag è già disponibile (es. già scatterato per un test precedente
    con lo stesso numero di partizioni) può essere passato per evitare di
    rifare lo scatter dei dati.
    """
    if X_bag is None:
        X_bag = _build_bag(client, X, num_partitions)

    clf = kmeans_parallel(k=k, l=l, r=r)

    try:
        start_time = time.time()
        clf.compute_starting_centroids(X_bag, seed=seed)
        clf.fit(X_bag, max_iter=max_iter_fit)
        elapsed_time = time.time() - start_time

        cost = calculate_inertia(X_bag, clf.final_centroids)
        print(f" -> Cost: {cost:.2f} | Time: {elapsed_time:.2f}s")
    except ValueError as e:
        # può succedere che il campionamento casuale non produca abbastanza
        # candidati centroidi (l troppo piccolo rispetto a k): logghiamo
        # il fallimento invece di interrompere l'intero benchmark.
        cost, elapsed_time = None, None
        print(f" -> FAILED: {e}")

    return {
        "k": k,
        "l": l,
        "r": r,
        "partitions": num_partitions,
        "cost": cost,
        "time": elapsed_time,
    }, X_bag


def run_benchmark(client, X, combinations, k_values, label="benchmark", max_iter_fit=10, seed=42,averaging_iterations = 10):
    """
    Esegue una griglia di test e salva i risultati in un CSV timestampato
    dentro ./results.

    Parameters
    ----------
    client : dask.distributed.Client
        Client connesso al cluster Dask.
    X : np.ndarray
        Dataset (già caricato/preprocessato) su cui testare.
    combinations : list of tuple
        Lista di (n_workers, num_partitions, l_over_k, r). n_workers è
        informativo/di logging: il cluster va già dimensionato a monte con
        launch_cluster(n_workers); qui serve solo a evitare di rifare lo
        scatter se non cambia rispetto alla run precedente.
    k_values : list of int
        Valori di k da testare; ogni combinazione viene ripetuta per ogni k.
    label : str
        Prefisso usato nel nome del file CSV di output.
    max_iter_fit : int
        Numero massimo di iterazioni di Lloyd's per la fase di fit().
    seed : int
        Seed per la riproducibilità del campionamento.
    averaging_iterations : int
        Numero di iterazioni su cui effettuare la media
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame con tutti i risultati.
    """
    results = []
    current_workers = None
    current_partitions = None
    X_bag = None

    for k in k_values:
        for n_workers, num_partitions, l_over_k, r in combinations:

            if n_workers != current_workers:
                current_workers = n_workers
                current_partitions = None

            if num_partitions != current_partitions:
                X_bag = _build_bag(client, X, num_partitions)
                current_partitions = num_partitions

            l = max(1, round(l_over_k * k))
            if l/k <= 0.1:
                r=15
                
            print(f"Testing: k={k}, workers={n_workers}, partitions={num_partitions}, "
                  f"l={l} (l/k={l_over_k}), r={r}",f"\n Iterating {averaging_iterations} times.")

            for i in range(averaging_iterations):
                result, X_bag = run_single_test(
                    client, X, k=k, l=l, r=r,
                    num_partitions=num_partitions,
                    max_iter_fit=max_iter_fit, seed=seed,
                    X_bag=X_bag,
                )
                result["workers"] = n_workers
                result["l_over_k"] = l_over_k
                results.append(result)

    df_results = pd.DataFrame(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"{label}_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nRisultati salvati in: {csv_path}")

    print("\n--- Benchmark Complete ---")
    for res in sorted(results, key=lambda x: (x["cost"] is None, x["cost"])):
        print(res)

    return df_results
