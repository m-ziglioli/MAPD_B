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

from src.kmeans_parallel import (
    _bag_to_matrices,
    _inertia_partial,
    kmeans_parallel,
)

RESULTS_DIR = "results"


def _build_bag(client, X, num_partitions):
    """Scatter dei dati ai worker e costruzione del Dask Bag dai futures
    (evita di far passare l'intero array attraverso il task graph)."""
    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)
    return db.from_delayed([dask.delayed(f) for f in futures])


def calculate_inertia(X_bag, centroids):
    """Inertia (somma delle d^2 al centroide piu' vicino) calcolata con un
    task vettorizzato per partizione: nessuna lambda punto-per-punto."""
    centroids_arr = np.vstack(centroids)
    partials = dask.compute(
        *[dask.delayed(_inertia_partial, pure=False)(p, centroids_arr)
          for p in _bag_to_matrices(X_bag)]
    )
    return float(sum(partials))


def run_single_test(client, k, l, r, num_partitions, max_iter_fit=10, seed=42, X=None, X_bag=None,
                     track_convergence=False, track_centroids=False):
    """
    Esegue una singola run di k-means|| + Lloyd's fit su una combinazione
    di parametri, e ritorna un dict con i risultati.

    Se X_bag è già disponibile (es. già scatterato per un test precedente
    con lo stesso numero di partizioni) può essere passato per evitare di
    rifare lo scatter dei dati.

    Se track_convergence=True, il dict di ritorno include anche
    "cost_history" e "iter_times" (uno per iterazione di Lloyd's fit),
    utili per ispezionare la convergenza di questa run.

    Se track_centroids=True, il dict di ritorno include anche
    "n_centroids_history" (numero cumulativo di candidati centroidi dopo
    ogni round di k-means||).
    """
    if X_bag is None:
        X_bag = _build_bag(client, X, num_partitions)

    clf = kmeans_parallel(k=k, l=l, r=r)
    cost_history, iter_times, n_centroids_history = None, None, None

    try:
        start_time = time.time()
        clf.compute_starting_centroids(X_bag, seed=seed, track_centroids=track_centroids)
        clf.fit(X_bag, max_iter=max_iter_fit, track_convergence=track_convergence)
        elapsed_time = time.time() - start_time

        cost = calculate_inertia(X_bag, clf.final_centroids)
        if track_convergence:
            cost_history, iter_times = clf.cost_history_, clf.iter_times_
        if track_centroids:
            n_centroids_history = clf.n_centroids_history_
        print(f" -> Cost: {cost:.2f} | Time: {elapsed_time:.2f}s")
    except ValueError as e:
        # può succedere che il campionamento casuale non produca abbastanza
        # candidati centroidi (l troppo piccolo rispetto a k): logghiamo
        # il fallimento invece di interrompere l'intero benchmark.
        cost, elapsed_time = None, None
        print(f" -> FAILED: {e}")

    result = {
        "k": k,
        "l": l,
        "r": r,
        "partitions": num_partitions,
        "cost": cost,
        "time": elapsed_time,
    }
    if track_convergence:
        result["cost_history"] = cost_history
        result["iter_times"] = iter_times
    if track_centroids:
        result["n_centroids_history"] = n_centroids_history

    return result, X_bag


def run_benchmark(client, X_bag, combinations, k_values, label="benchmark", max_iter_fit=10, seed=42,averaging_iterations = 10):
    """
    Esegue una griglia di test e salva i risultati in un CSV timestampato
    dentro ./results.

    Parameters
    ----------
    client : dask.distributed.Client
        Client connesso al cluster Dask.
    X_bag : dask.bag.Bag
        Dataset (già caricato/preprocessato) su cui testare, come Dask Bag
        distribuita (es. l'output di data_loader.load_dataset). Viene
        materializzato in un array numpy lato client una sola volta
        all'inizio (costo limitato alla dimensione del dataset, pagato una
        volta per l'intero benchmark), poi per ogni combinazione viene
        ridistribuito ai worker con client.scatter() (via _build_bag) al
        numero di partizioni richiesto. Bag.repartition() è stato scartato:
        nei test si è rivelato un collo di bottiglia poco parallelizzato
        (vedi CHANGES.md), mentre client.scatter() distribuisce bene su
        tutto il cluster.
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
    # Materializziamo il dataset lato client una sola volta. X_bag contiene
    # un array numpy separato per ogni singola riga (vedi data_loader.py),
    # quindi un gather diretto dovrebbe serializzare ~500k oggetti minuscoli
    # invece di poche decine di blocchi compatti: qui vstack-iamo ogni
    # partizione in UN SOLO array 2D dentro il grafo Dask (calcolato sul
    # worker), cosi' sulla rete viaggiano solo len(partitions) blocchi
    # compatti invece di centinaia di migliaia di piccoli oggetti.
    delayed_partitions = [dask.delayed(np.vstack)(p) for p in X_bag.to_delayed()]
    partition_futures = client.compute(delayed_partitions)
    partitions = client.gather(partition_futures)
    X_arr = np.vstack(partitions)

    results = []
    current_workers = None
    current_partitions = None
    current_bag = None

    for k in k_values:
        for n_workers, num_partitions, l_over_k, r in combinations:

            if n_workers != current_workers:
                current_workers = n_workers
                current_partitions = None

            if num_partitions != current_partitions:
                old_bag = current_bag
                current_bag = _build_bag(client, X_arr, num_partitions)
                if old_bag is not None:
                    client.cancel(old_bag)
                current_partitions = num_partitions

            l = max(1, round(l_over_k * k))
            if l/k <= 0.1:
                r=15

            print(f"Testing: k={k}, workers={n_workers}, partitions={num_partitions}, "
                  f"l={l} (l/k={l_over_k}), r={r}",f"\n Iterating {averaging_iterations} times.")

            for i in range(averaging_iterations):
                result, current_bag = run_single_test(
                    client, k=k, l=l, r=r,
                    num_partitions=num_partitions,
                    max_iter_fit=max_iter_fit, seed=seed,
                    X_bag=current_bag,
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
