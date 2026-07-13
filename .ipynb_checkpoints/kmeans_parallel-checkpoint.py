"""
kmeans_parallel.py
===================
Implementazione dell'algoritmo di inizializzazione k-means|| (parallel
k-means++) su Dask Bag, seguita da una standard Lloyd's iteration (fit)
distribuita.

Uso tipico (da notebook):

    from kmeans_parallel import kmeans_parallel

    clf = kmeans_parallel(k=500, l=200, r=5)
    clf.compute_starting_centroids(X_bag, seed=42)
    clf.fit(X_bag, max_iter=10)
    labels = clf.classify(X_bag)
"""

import dask
import dask.bag as db
from sklearn.cluster import KMeans
import numpy as np
from dask.bag import random as dbr

class kmeans_parallel():
    """K-means con inizializzazione parallela (k-means||) su Dask."""

    # --------------------------------------------------------------------

    def __init__(self, k, l, r):
        """
        Parameters
        ----------
        k : int
            Numero di cluster finali desiderati.
        l : int
            Fattore di oversampling: numero atteso di candidati campionati
            ad ogni round dell'inizializzazione parallela.
        r : int
            Numero di round dell'inizializzazione parallela (se None,
            viene stimato automaticamente da compute_starting_centroids).
        """
        self.k = k
        self.l = l  # oversampling factor
        self.r = r  # number of iterations
        self.centroids = []

    # --------------------------------------------------------------------

    def compute_starting_centroids(self, X, alpha=1, l=None, max_iter=None, seed=None):
        """Inizializzazione k-means|| parallela: seleziona un pool di
        candidati centroidi campionando iterativamente da X con
        probabilità proporzionale alla distanza al quadrato dal centroide
        più vicino già scelto, poi li riduce a k centroidi finali con un
        k-means pesato (scikit-learn)."""

        if seed is not None:
            np.random.seed(seed)

        # STEP 1: centroide iniziale casuale
        initial_centroid = dbr.sample(X, 1).compute()[0]
        initial_centroid = np.asarray(initial_centroid).reshape(1, -1)
        self.centroids.append(initial_centroid)

        c0 = initial_centroid[0]
        # bag di (distanza minima al quadrato dal cluster più vicino, indice cluster più vicino)
        state = X.map(lambda x: (np.linalg.norm(x - c0) ** 2, 0))
        state = dask.persist(state)[0]

        # STEP 2: costo iniziale
        psi = state.map(lambda t: t[0]).sum().compute()

        # STEP 3: determina il numero di round
        if l is None:
            l = self.l
        if max_iter is None:
            max_iter = self.r if self.r is not None else int(round(alpha * np.log(psi)))

        for _ in range(max_iter):
            # costo corrente
            cost = state.map(lambda t: t[0]).sum().compute()

            # probabilità di campionamento per ogni punto
            probs = state.map(lambda t: min(1.0, t[0] * l / cost))

            # accoppia punti e probabilità, campiona con Bernoulli(prob)
            paired = db.zip(X, probs)
            sample = (
                paired.filter(lambda t: np.random.uniform() < t[1])
                      .map(lambda t: t[0])
            ).compute()

            if not sample:
                continue

            start_idx = len(self.centroids)
            new_centroids_arr = np.vstack([np.asarray(s).reshape(1, -1) for s in sample])

            for s in sample:
                self.centroids.append(np.asarray(s).reshape(1, -1))

            def update_state(pack, new_arr=new_centroids_arr, s_idx=start_idx):
                x, (curr_min_dist, curr_idx) = pack
                dists_to_new = np.linalg.norm(x - new_arr, axis=1) ** 2
                min_new_idx_relative = int(np.argmin(dists_to_new))
                min_new_dist = dists_to_new[min_new_idx_relative]

                if min_new_dist < curr_min_dist:
                    return (min_new_dist, s_idx + min_new_idx_relative)
                else:
                    return (curr_min_dist, curr_idx)

            state = db.zip(X, state).map(update_state)
            state = dask.persist(state)[0]

        # STEP 7: pesi = numero di punti assegnati a ciascun candidato centroide
        counts = state.map(lambda t: t[1]).frequencies().compute()
        weights = np.zeros(len(self.centroids))

        for center_idx, count in counts:
            weights[center_idx] = count

        centroids_weights = weights

        # STEP 8: riduzione finale a k centroidi con k-means pesato (scikit-learn)
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(np.vstack(self.centroids), sample_weight=centroids_weights)
        self.starting_centroids = kmeans.cluster_centers_

        self.min_dists = state.map(lambda t: t[0])

    # --------------------------------------------------------------------

    def fit(self, X, max_iter=100, tol=1e-4):
        """Standard Lloyd's K-means, a partire dai centroidi calcolati da
        compute_starting_centroids, eseguito in modo distribuito su X."""

        centroids_arr = np.vstack(self.starting_centroids)

        for _ in range(max_iter):
            mapped = X.map(lambda x: (int(np.argmin(np.linalg.norm(x - centroids_arr, axis=1))), np.array(x)))

            def binop(acc, item):
                total, count = acc
                return (total + item[1], count + 1)

            def combine(acc1, acc2):
                return (acc1[0] + acc2[0], acc1[1] + acc2[1])

            sums_counts = mapped.foldby(lambda t: t[0], binop, (0.0, 0), combine, (0.0, 0)).compute()
            reduced = {idx: total / count for idx, (total, count) in sums_counts}

            new_centroids = np.copy(centroids_arr)
            for cluster_idx, p_mean in reduced.items():
                new_centroids[cluster_idx] = p_mean

            self.final_centroids = new_centroids

            if np.linalg.norm(new_centroids - centroids_arr) < tol:
                break

            centroids_arr = new_centroids

    # --------------------------------------------------------------------

    def classify(self, X):
        """Ritorna un Dask Bag con l'indice del cluster più vicino per
        ogni punto di X, usando i centroidi finali calcolati da fit()."""
        centroids_arr = np.vstack(self.final_centroids)
        return X.map(lambda x: int(np.argmin(np.linalg.norm(x - centroids_arr, axis=1))))

    # --------------------------------------------------------------------

    def inertia(self, X):
        """Calcola l'inertia (somma delle distanze al quadrato dai
        centroidi finali) sull'intero dataset X."""
        centroids_arr = np.vstack(self.final_centroids)
        return X.map(lambda x: np.min(np.linalg.norm(x - centroids_arr, axis=1) ** 2)).sum().compute()
