"""
kmeans_parallel.py
==================
Implementazione dell'algoritmo di inizializzazione k-means|| (parallel
k-means++) su Dask Bag, seguita da una standard Lloyd's iteration (fit)
distribuita.

Motore computazionale (refactor 2026-08-23)
-------------------------------------------
Le operazioni punto-per-punto (lambda sulla singola riga) sono state
sostituite da calcoli vettorizzati PER PARTIZIONE: la bag viene scomposta
in partizioni ritardate (``X.to_delayed()``), ognuna impilata in una
matrice densa ``(m, d)``, e ogni passaggio dell'algoritmo diventa un task
per partizione che restituisce solo le riduzioni necessarie (somme per
cluster, conteggi, costo, etichette cambiate). Rispetto all'implementazione
precedente questo elimina milioni di chiamate Python e lo shuffle di righe
del foldby: attraverso i confini di partizione viaggiano solo matrici k x d.

Il seeding e' deterministico: tutte le estrazioni (centroide iniziale,
Bernoulli dei round, random_state del reclustering pesato) derivano da
SeedSequence(seed), con RNG per (partizione, round) — l'ordine di esecuzione
dello scheduler non influenza piu' il risultato (docs/CHANGES.md 2026-07-19).

Uso tipico (da notebook):

    from src.kmeans_parallel import kmeans_parallel

    clf = kmeans_parallel(k=500, l=200, r=5)
    clf.compute_starting_centroids(X_bag, seed=42)
    clf.fit(X_bag, max_iter=10)
    labels = clf.classify(X_bag)
"""

import time
import warnings

import dask
import dask.bag as db
from sklearn.cluster import KMeans
import numpy as np


# ----------------------------------------------------------------------------
# Helper vettorizzati per-partizione (funzioni module-level: serializzabili
# dai scheduler senza closure complesse)
# ----------------------------------------------------------------------------

def _stack_rows(rows):
    """Lista di righe 1-D -> matrice densa (m, d) float64."""
    arrs = [np.asarray(r, dtype=np.float64) for r in rows]
    if not arrs:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(arrs)


def _bag_to_matrices(X):
    """Bag -> lista di Delayed, uno per partizione, ognuno una matrice (m, d)."""
    return [dask.delayed(_stack_rows, pure=True)(p) for p in X.to_delayed()]


def _matrix_shape(M):
    return M.shape


def _pairwise_d2(M, C):
    """d2[i, j] = ||M[i] - C[j]||^2 con l'espansione quadratica (BLAS-friendly).

    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c : la distanza fra tutte le righe
    e tutti i centroidi e' un'unica matmul. Il clipping a 0 evita piccoli
    valori negativi dovuti alla aritmetica floating-point.
    """
    m_sq = np.einsum("ij,ij->i", M, M)
    # ->i: una norma per CENTROIDE (riga di C). Con ->j si otterrebbero le
    # somme per colonna (d valori) e il broadcasting con (M @ C.T) (m, t)
    # fallirebbe o, se t == d, produrrebbe distanze sbagliate in silenzio.
    c_sq = np.einsum("ij,ij->i", C, C)
    d2 = m_sq[:, None] + c_sq[None, :] - 2.0 * (M @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _lloyd_pass(M, prev_labels, C):
    """Una iterazione di assegnazione Lloyd's su una partizione.

    Ritorna (sums (k,d), counts (k,), cost, changed, labels (m,) int32).
    ``changed`` confronta le etichette con quelle dell'iterazione precedente
    (fusa nel passaggio: nessun giro extra sui dati per il criterio di
    convergenza stretto).
    """
    k = C.shape[0]
    if M.shape[0] == 0:
        return (np.zeros((k, C.shape[1])), np.zeros(k, dtype=np.int64),
                0.0, 0, np.empty(0, dtype=np.int32))
    d2 = _pairwise_d2(M, C)
    labels = d2.argmin(axis=1)
    sums = np.zeros((k, M.shape[1]))
    np.add.at(sums, labels, M)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    cost = float(d2[np.arange(M.shape[0]), labels].sum())
    if prev_labels is None:
        changed = int(M.shape[0])
    else:
        changed = int((labels != prev_labels).sum())
    return sums, counts, cost, changed, labels.astype(np.int32)


def _init_state(M, c0):
    """Stato iniziale k-means|| per partizione: colonna 0 = d^2(x, c0),
    colonna 1 = indice del centroide piu' vicino (0)."""
    if M.shape[0] == 0:
        return np.empty((0, 2))
    d2 = _pairwise_d2(M, c0.reshape(1, -1))[:, 0]
    return np.column_stack([d2, np.zeros(len(M))])


def _sample_round(M, state, l, cost, seed_seq, round_idx):
    """Campionamento Bernoulli vettorizzato su una partizione.

    Ogni punto viene campionato con probabilita' min(1, l * d2 / cost) usando
    un RNG locale derivato da (seed_seq, indice di partizione implicito nel
    seed_seq figlio, round): deterministico e parallelo-safe.
    Ritorna la matrice (t, d) dei punti campionati.
    """
    if M.shape[0] == 0 or cost <= 0.0:
        return np.empty((0, M.shape[1] if M.ndim == 2 else 0))
    rng = np.random.default_rng([seed_seq.entropy % (2**32), *seed_seq.spawn_key, round_idx])
    probs = np.minimum(1.0, state[:, 0] * l / cost)
    mask = rng.random(M.shape[0]) < probs
    return M[mask]


def _update_state(M, state, new_centroids, start_idx):
    """Aggiorna lo stato (d^2 min, indice centroide) dopo aver aggiunto i
    candidati ``new_centroids`` (t, d) che partono dall'indice start_idx."""
    if M.shape[0] == 0 or new_centroids.shape[0] == 0:
        return state
    d2_new = _pairwise_d2(M, new_centroids)
    min_new_idx = d2_new.argmin(axis=1)
    min_new_dist = d2_new[np.arange(M.shape[0]), min_new_idx]
    closer = min_new_dist < state[:, 0]
    out = state.copy()
    out[closer, 0] = min_new_dist[closer]
    out[closer, 1] = (min_new_idx[closer] + start_idx).astype(state[:, 1].dtype)
    return out


def _partition_bincount(state, n_centers):
    """Istogramma delle assegnazioni (pesi del reclustering) per partizione."""
    if state.shape[0] == 0:
        return np.zeros(n_centers, dtype=np.int64)
    return np.bincount(state[:, 1].astype(np.int64), minlength=n_centers)


def _labels_partition(M, C):
    """Etichette per partizione, come lista di int scalari (semantica Bag)."""
    if M.shape[0] == 0:
        return []
    return _pairwise_d2(M, C).argmin(axis=1).tolist()


def _inertia_partial(M, C):
    """Somma parziale delle d^2 al centroide piu' vicino, per partizione."""
    if M.shape[0] == 0:
        return 0.0
    d2 = _pairwise_d2(M, C)
    return float(d2[np.arange(M.shape[0]), d2.argmin(axis=1)].sum())


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
        self.n_iter_ = None  # iterazioni di Lloyd's eseguite da fit()

    # --------------------------------------------------------------------

    def compute_starting_centroids(self, X, alpha=1, l=None, max_iter=None, seed=None, track_centroids=False):
        """Inizializzazione k-means|| parallela: seleziona un pool di
        candidati centroidi campionando iterativamente da X con
        probabilita' proporzionale alla distanza al quadrato dal centroide
        piu' vicino gia' scelto, poi li riduce a k centroidi finali con un
        k-means pesato (scikit-learn).

        Se track_centroids=True, salva in self.n_centroids_history_ il
        numero cumulativo di candidati centroidi dopo ogni round eseguito,
        utile per ispezionare la crescita del pool di candidati.

        Deterministico dato ``seed``: le estrazioni usano SeedSequence(seed)
        e un RNG per (partizione, round), non il RNG globale di NumPy.
        """
        parts = _bag_to_matrices(X)
        shapes = dask.compute(*[dask.delayed(_matrix_shape, pure=True)(p) for p in parts])
        n_points = int(sum(s[0] for s in shapes))

        if track_centroids:
            self.n_centroids_history_ = []

        # SeedSequence padre: entropia deterministica se seed e' fornito,
        # casuale altrimenti. Da qui derivano TUTTE le estrazioni.
        seed_seq = np.random.SeedSequence(seed)

        # STEP 1: centroide iniziale casuale (uniforme su tutti i punti,
        # via indice estratto dal RNG dedicato)
        rng = np.random.default_rng(seed_seq)
        initial_idx = int(rng.integers(n_points))
        offset = 0
        for p_idx, (m, _) in enumerate(shapes):
            if initial_idx < offset + m:
                local = initial_idx - offset
                initial_centroid = np.asarray(
                    dask.delayed(lambda M, i: M[i].copy(), pure=True)(
                        parts[p_idx], local
                    ).compute(),
                    dtype=np.float64,
                )
                break
            offset += m
        initial_centroid = initial_centroid.reshape(1, -1)
        self.centroids.append(initial_centroid)

        # stato per partizione: (m, 2) -> (d^2 minima, indice centroide)
        states = dask.compute(
            *[dask.delayed(_init_state, pure=True)(p, initial_centroid[0]) for p in parts]
        )

        # STEP 2: costo iniziale
        psi = float(sum(s[:, 0].sum() for s in states))
        if psi == 0.0:
            # Tutti i punti coincidono con il centroide iniziale: non c'e'
            # nulla da campionare. Si rispetta il contratto della classe
            # (starting_centroids ha esattamente k righe) ripetendo il
            # centroide; il costo e' comunque 0.
            self.starting_centroids = np.repeat(initial_centroid, self.k, axis=0)
            return

        # STEP 3: determina il numero di round
        if l is None:
            l = self.l
        if max_iter is None:
            ratio = (l if l is not None else self.l) / self.k
            if ratio <= 0.1:
                max_iter = 15
                # se l/k è piccolo, per piccoli valori di r rischiamo di avere meno di k valori, e il codice si interrompe
            elif self.r is not None:
                max_iter = self.r
            else:
                max_iter = int(round(alpha * np.log(psi)))

        # chiavi RNG figlie, una per partizione: indipendenti per costruzione
        child_seeds = seed_seq.spawn(len(parts))

        cost = psi
        for round_idx in range(max_iter):
            if cost == 0.0:
                break

            # probabilità di campionamento per ogni punto (vettore per partizione)
            sample_tasks = [
                dask.delayed(_sample_round, pure=False)(
                    p, s, l, cost, child_seeds[j], round_idx
                )
                for j, (p, s) in enumerate(zip(parts, states))
            ]
            sampled_parts = dask.compute(*sample_tasks)
            sampled = [s for s in sampled_parts if s.shape[0] > 0]

            if sampled:
                new_centroids_arr = np.vstack(sampled)
                start_idx = len(self.centroids)
                self.centroids.extend(
                    row.reshape(1, -1) for row in new_centroids_arr
                )

                update_tasks = [
                    dask.delayed(_update_state, pure=False)(
                        p, s, new_centroids_arr, start_idx
                    )
                    for p, s in zip(parts, states)
                ]
                states = dask.compute(*update_tasks)

            if track_centroids:
                self.n_centroids_history_.append(len(self.centroids))

            # aggiorno il costo corrente per il prossimo round
            cost = float(sum(s[:, 0].sum() for s in states))

        # STEP 7: pesi = numero di punti assegnati a ciascun candidato centroide
        weights = sum(
            dask.compute(*[dask.delayed(_partition_bincount, pure=True)(s, len(self.centroids))
                           for s in states])
        )
        centroids_weights = weights.astype(np.float64)

        # STEP 8: riduzione finale a k centroidi con k-means pesato (scikit-learn)
        # n_init=1: il paper (Bahmani et al.) usa una singola inizializzazione
        # k-means++ per il reclustering, non i 10 restart di default di sklearn.
        # random_state deriva dalla stessa SeedSequence del seeding: senza,
        # il k-means++ interno di sklearn pescherebbe dal RNG globale e il
        # risultato non sarebbe riproducibile nemmeno a parita' di seed.
        reclustering_random_state = int(np.random.default_rng(seed_seq).integers(2**31 - 1))
        kmeans = KMeans(n_clusters=self.k, n_init=1, random_state=reclustering_random_state)
        kmeans.fit(np.vstack(self.centroids), sample_weight=centroids_weights)
        self.starting_centroids = kmeans.cluster_centers_

    # --------------------------------------------------------------------

    def fit(self, X, max_iter=100, tol=1e-4, track_convergence=False):
        """Standard Lloyd's K-means, a partire dai centroidi calcolati da
        compute_starting_centroids, eseguito in modo distribuito su X.

        Un unico percorso di codice: ogni iterazione esegue un task per
        partizione che calcola assegnazioni, somme per cluster, costo e
        numero di etichette cambiate (criterio di convergenza stretto fuso
        nello stesso passaggio: nessun giro extra sui dati). Se
        track_convergence=True, le quantita' vengono anche registrate in
        self.cost_history_ (inertia ad ogni iterazione) e self.iter_times_
        (tempo per iterazione).

        Al termine, self.n_iter_ contiene il numero di iterazioni di
        Lloyd's effettivamente eseguite (aggiornamenti dei centroidi
        completati).
        """

        parts = _bag_to_matrices(X)
        n_partitions = len(parts)

        centroids_arr = np.vstack(self.starting_centroids)
        k = centroids_arr.shape[0]

        if track_convergence:
            self.cost_history_ = []
            self.iter_times_ = []

        prev_labels = [None] * n_partitions
        empty_warned = False

        for iteration in range(max_iter):
            iter_start = time.time()

            tasks = [
                dask.delayed(_lloyd_pass, pure=False)(p, prev_labels[j], centroids_arr)
                for j, p in enumerate(parts)
            ]
            results = dask.compute(*tasks)

            sums = np.zeros_like(centroids_arr)
            counts = np.zeros(k, dtype=np.int64)
            iter_cost = 0.0
            changed = 0
            for j, (p_sums, p_counts, p_cost, p_changed, p_labels) in enumerate(results):
                sums += p_sums
                counts += p_counts
                iter_cost += p_cost
                changed += p_changed
                prev_labels[j] = p_labels

            new_centroids = centroids_arr.copy()
            populated = counts > 0
            new_centroids[populated] = sums[populated] / counts[populated, None]
            if not empty_warned and not populated.all():
                warnings.warn(
                    f"{int((~populated).sum())} cluster vuoti dopo l'assegnazione: "
                    "i centroidi corrispondenti restano al valore dell'iterazione precedente"
                )
                empty_warned = True

            self.final_centroids = new_centroids
            # numero di update di Lloyd's completati
            self.n_iter_ = iteration + 1

            if track_convergence:
                self.cost_history_.append(iter_cost)
                self.iter_times_.append(time.time() - iter_start)

            # Strict convergence: stop as soon as no point changes cluster,
            # like sklearn does. The raw centroid-shift check below almost
            # never triggers once k is in the hundreds (its threshold does
            # not scale with k), so without this check fit() runs until
            # max_iter even when the assignment has long been stable.
            if changed == 0:
                break

            if np.linalg.norm(new_centroids - centroids_arr) < tol:
                break

            centroids_arr = new_centroids

    # --------------------------------------------------------------------

    def classify(self, X):
        """Ritorna un Dask Bag con l'indice del cluster più vicino per
        ogni punto di X, usando i centroidi finali calcolati da fit()."""
        centroids_arr = np.vstack(self.final_centroids)
        label_delays = [
            dask.delayed(_labels_partition, pure=False)(p, centroids_arr)
            for p in _bag_to_matrices(X)
        ]
        return db.from_delayed(label_delays)

    # --------------------------------------------------------------------

    def inertia(self, X):
        """Calcola l'inertia (somma delle distanze al quadrato dai
        centroidi finali) sull'intero dataset X."""
        centroids_arr = np.vstack(self.final_centroids)
        partials = dask.compute(
            *[dask.delayed(_inertia_partial, pure=False)(p, centroids_arr)
              for p in _bag_to_matrices(X)]
        )
        return float(sum(partials))
