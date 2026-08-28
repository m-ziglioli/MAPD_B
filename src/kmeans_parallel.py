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


def _persist_matrices(X):
    """Come _bag_to_matrices, ma le matrici per partizione vengono
    MATERIALIZZATE una sola volta (futures sul cluster, cache nello
    scheduler locale). Tutti i task successivi di seeding/fit referenziano
    questi nodi: senza persistenza ogni round/iterazione rimpilerebbe da
    capo le partizioni del bag (vstack di migliaia di righe minuscole),
    moltiplicando inutilmente il lavoro piu' costoso del motore."""
    parts = _bag_to_matrices(X)
    if parts:
        parts = list(dask.persist(*parts))
    return parts


def _matrix_shape(M):
    return M.shape


def _row_at(M, i):
    """Riga i-esima di una matrice per partizione (copia indipendente)."""
    return M[i].copy()


def _state_cost(state):
    """Somma delle d^2 minime correnti su una partizione (scalare)."""
    return float(state[:, 0].sum())


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

# NOTE: kmeans parallel code is written such that each machine computes distances etc locally, using numpy etc (no dask subtasks) - this is ok because dataset small and dask approach would probably add overhead (aside from being more complex).
# Main problem was that materialising full distance matrix on a single machine leads to crashing: for 5 workers (8 partitions per worker), k=100, 4e6 total points, we have 1e7 elements, each 8 Bytes, 8 cores simultaneously ---> 640 MB; then numpy operations can require 3x that ---> about 2 GB more.
def _pairwise_d2_argmin_chunked(M, C, chunk_k=20):
    """
    Calcola, per ogni punto in M, la distanza al quadrato dal centroide
    più vicino in C - SENZA mai costruire la matrice completa (m, k).

    Perché serve: la versione originale (_pairwise_d2) calcola tutte le
    distanze in un colpo solo, creando una matrice di dimensione
    (numero di punti) x (numero di centroidi). Quando k è grande
    (es. 500-1000) e la partizione è grande, questa matrice puo' pesare
    diversi GB in memoria - questo e' probabilmente il bug che causava gli out-of-memory.

    Come funziona: invece di guardare tutti i k centroidi insieme,
    li guardiamo a piccoli gruppi ("chunk") per volta. Per ogni gruppo,
    calcoliamo le distanze solo verso quel gruppo, e teniamo traccia del
    miglior risultato visto finora (distanza minima e indice del centroide
    corrispondente). Alla fine otteniamo lo stesso risultato di prima, ma
    non abbiamo mai tenuto in memoria più di (m, chunk_k) numeri insieme.

    Parametri
    ---------
    M : array (m, d)
        I punti della partizione (m righe, d feature per riga).
    C : array (k, d)
        Tutti i centroidi attuali (k centroidi, d feature ciascuno).
    chunk_k : int
        Quanti centroidi considerare per volta. Più piccolo = meno memoria
        usata ma più iterazioni del ciclo (leggermente più lento).
        Più grande = più memoria ma meno iterazioni.

    Ritorna
    -------
    best_dist : array (m,)
        Per ogni punto, la distanza al quadrato dal centroide più vicino.
    best_idx : array (m,) di interi
        Per ogni punto, l'indice (0-based) del centroide più vicino.
    """
    m = M.shape[0]  # numero di punti in questa partizione

    # Se la partizione e' vuota, non c'e' nulla da calcolare.
    if m == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)

    # Qui memorizziamo il MIGLIOR risultato trovato finora per ogni punto.
    # Iniziamo con "infinito" come distanza (cosi' il primo chunk vince
    # sicuramente) e indice 0 come segnaposto.
    best_dist = np.full(m, np.inf)
    best_idx = np.zeros(m, dtype=np.int64)

    # ||x||^2 per ogni punto x in M. Lo calcoliamo una sola volta fuori dal
    # ciclo perche' non cambia mai (dipende solo da M, non dal centroide).
    m_sq = np.einsum("ij,ij->i", M, M)

    # Scorriamo i centroidi a gruppi di "chunk_k" alla volta.
    # Esempio: se k=1000 e chunk_k=100, questo ciclo gira 10 volte.
    for start in range(0, C.shape[0], chunk_k):
        # Prendiamo solo il pezzo di centroidi che ci interessa in questo giro.
        end = start + chunk_k
        C_chunk = C[start:end]  # forma: (fino a chunk_k, d)

        # ||c||^2 per ogni centroide in questo chunk.
        c_sq = np.einsum("ij,ij->i", C_chunk, C_chunk)

        # Distanza al quadrato fra ogni punto di M e ogni centroide del chunk:
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x . c)
        # Questa matrice ha forma (m, chunk_k) — MOLTO più piccola della
        # matrice completa (m, k) che causava il problema di memoria.
        d2_chunk = m_sq[:, None] + c_sq[None, :] - 2.0 * (M @ C_chunk.T)

        # Piccoli errori di arrotondamento float possono dare valori
        # leggermente negativi invece di 0: li correggiamo.
        np.maximum(d2_chunk, 0.0, out=d2_chunk)

        # Per ogni punto, qual e' il centroide migliore DENTRO QUESTO CHUNK?
        local_best_idx = d2_chunk.argmin(axis=1)          # indice locale (0..chunk_k-1)
        local_best_dist = d2_chunk[np.arange(m), local_best_idx]  # distanza corrispondente

        # Confrontiamo con il migliore trovato finora (nei chunk precedenti).
        # "is_better" e' True per i punti dove QUESTO chunk ha un centroide
        # piu' vicino di quanto trovato prima.
        is_better = local_best_dist < best_dist

        # Aggiorniamo solo i punti per cui abbiamo trovato qualcosa di meglio.
        best_dist[is_better] = local_best_dist[is_better]
        # Attenzione: local_best_idx e' relativo al chunk (parte da 0),
        # quindi dobbiamo sommare "start" per ottenere l'indice vero nel
        # vettore completo dei k centroidi.
        best_idx[is_better] = local_best_idx[is_better] + start

    return best_dist, best_idx


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

        
    #d2 = _pairwise_d2(M, C)
    #labels = d2.argmin(axis=1)
    #sums = np.zeros((k, M.shape[1]))
    #np.add.at(sums, labels, M)
    #counts = np.bincount(labels, minlength=k).astype(np.int64)
    #cost = float(d2[np.arange(M.shape[0]), labels].sum())

    # Usiamo la versione "a chunk" per non costruire mai la matrice
    # completa (m, k) delle distanze, che 
    # puo' occupare diversi GB e causare out-of-memory.
    best_dist, labels = _pairwise_d2_argmin_chunked(M, C)
    sums = np.zeros((k, M.shape[1]))
    np.add.at(sums, labels, M)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    # best_dist contiene gia' la distanza al centroide assegnato per ogni
    # punto: la sommiamo direttamente, senza dover rileggere da d2.
    cost = float(best_dist.sum())
    
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


def _sample_round(M, state, l, cost, round_seed_seq):
    """Campionamento Bernoulli vettorizzato su una partizione.

    Ogni punto viene campionato con probabilita' min(1, l * d2 / cost) usando
    un RNG locale derivato dalla SeedSequence figlia (partizione, round),
    pre-derivata dal chiamante: deterministico e parallelo-safe, senza
    condivisione di stream fra estrazioni diverse.
    Ritorna la matrice (t, d) dei punti campionati.
    """
    if M.shape[0] == 0 or cost <= 0.0:
        return np.empty((0, M.shape[1] if M.ndim == 2 else 0))
    rng = np.random.default_rng(round_seed_seq)
    probs = np.minimum(1.0, state[:, 0] * l / cost)
    mask = rng.random(M.shape[0]) < probs
    return M[mask]


def _sample_round_exact(M, state, l, round_seed_seq):
    """Campionamento ESATTO di l punti per round, senza reimmissione, con
    probabilita' proporzionale a d^2(x, C) — protocollo della Fig 5.1 del
    paper (Bahmani et al. usano questa variante SOLO per la Fig 5.1).

    Schema Efraimidis-Spirakis (weighted sampling without replacement):
    chiave u^(1/w) con u~U(0,1), w = d^2; i top-l LOCALI di ogni partizione,
    uniti, contengono il top-l GLOBALE (proprieta' del massimo: top-l di un
   'unione = unione dei top-l). Cosi' attraverso il confine client/cluster
    passano solo l coppie (chiave, indice) per partizione, mai i punti.

    Punti a distanza zero hanno peso nullo e non sono mai campionabili
    (coerente con il Bernoulli, dove p = min(1, l*d2/cost) = 0).

    Ritorna (keys (t,), local_indices (t,) int64): le chiavi servono al
    merge globale sul client, gli indici al recupero delle righe scelte.
    """
    if M.shape[0] == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(round_seed_seq)
    u = rng.random(M.shape[0])
    w = state[:, 0].astype(np.float64)
    nz = w > 0.0
    # chiave -inf => punto non campionabile (peso nullo). Forma potenza
    # (non exp/log): u=0 esatto non genera warning e da' chiave 0.
    keys = np.full(M.shape[0], -np.inf)
    keys[nz] = u[nz] ** (1.0 / w[nz])
    l_loc = int(min(l, nz.sum()))
    if l_loc <= 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    top = np.argsort(keys)[-l_loc:]
    return keys[top], top.astype(np.int64)


def _rows_at(M, idx):
    """Righe di una partizione agli indici locali ``idx`` (fancy indexing);
    usato sia dal percorso r=0 sia dal recupero dei campioni esatti."""
    if len(idx) == 0:
        return np.empty((0, M.shape[1] if M.ndim == 2 else 0))
    return M[idx]


def _update_state(M, state, new_centroids, start_idx):
    """Aggiorna lo stato (d^2 min, indice centroide) dopo aver aggiunto i
    candidati ``new_centroids`` (t, d) che partono dall'indice start_idx.

    Ritorna (stato aggiornato, somma parziale delle d^2 minime): il costo
    del round e' fuso nell'aggiornamento, cosicche' attraverso il confine
    client/cluster passa solo lo scalare e non la matrice di stato.
    """
    # for debugging:
    print(f"M: {M.shape}, {M.nbytes/1e6:.1f} MB | "
          f"state: {state.shape}, {state.nbytes/1e6:.1f} MB | "
          f"new_centroids: {new_centroids.shape}, {new_centroids.nbytes/1e6:.1f} MB")
    
    if M.shape[0] == 0 or new_centroids.shape[0] == 0:
        return state, float(state[:, 0].sum())
    best_dist, best_idx = _pairwise_d2_argmin_chunked(M, new_centroids)
    closer = best_dist < state[:, 0]
    out = state.copy()
    out[closer, 0] = best_dist[closer]
    out[closer, 1] = (best_idx[closer] + start_idx).astype(state[:, 1].dtype)
    return out, float(out[:, 0].sum())


def _partition_bincount(state, n_centers):
    """Istogramma delle assegnazioni (pesi del reclustering) per partizione."""
    if state.shape[0] == 0:
        return np.zeros(n_centers, dtype=np.int64)
    return np.bincount(state[:, 1].astype(np.int64), minlength=n_centers)


def resolve_rounds(l, k, r=None, alpha=1.0, psi=None, policy="auto"):
    """Numero di round k-means|| da eseguire, in un UNICO punto del codice.

    policy="auto" (default) segue il protocollo del paper (Bahmani et al.,
    VLDB 2012):
      - se l/k <= 0.1 servono piu' round per accumulare almeno k candidati:
        vengono usati 15 round, a prescindere da ``r``;
      - altrimenti vince un ``r`` esplicito;
      - senza ``r`` si stima con round(alpha * log(psi)).

    policy="fixed" usa sempre ed esclusivamente ``r`` (obbligatorio):
    escape hatch esplicito per bypassare la regola del paper. Con r=0 il
    chiamante (compute_starting_centroids) ottiene la modalita' "random
    baseline": k centri uniformi senza round ne' reclustering.

    Nota: la regola l/k<=0.1 -> 15 e' volutamente mantenuta anche quando
    ``r`` e' fornito (comportamento storico del progetto); i driver
    registrano comunque il numero di round EFFETTIVO (``n_rounds_`` /
    colonna ``r_effective`` nei CSV) cosicche' le sweep confrontano le
    configurazioni sul valore realmente eseguito.
    """
    if policy not in ("auto", "fixed"):
        raise ValueError("policy deve essere 'auto' o 'fixed'")
    if policy == "fixed":
        if r is None:
            raise ValueError("policy='fixed' richiede un numero di round r esplicito")
        if int(r) < 0:
            raise ValueError("r non puo' essere negativo")
        return int(r)
    # policy == "auto"
    if l / k <= 0.1:
        return 15
    if r is not None:
        if int(r) < 0:
            raise ValueError("r non puo' essere negativo")
        return int(r)
    if psi is None or psi <= 0:
        raise ValueError("senza r esplicito serve psi > 0 per stimare i round")
    return max(1, int(round(alpha * float(np.log(psi)))))


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


def inertia_of_bag(X_bag, centroids):
    """Inertia (somma delle d^2 al centroide piu' vicino) sull'intera bag,
    con un task vettorizzato per partizione. Unico punto di calcolo condiviso
    da kmeans_parallel.inertia() e benchmark.calculate_inertia()."""
    centroids_arr = np.vstack(centroids)
    partials = dask.compute(
        *[dask.delayed(_inertia_partial, pure=False)(p, centroids_arr)
          for p in _bag_to_matrices(X_bag)]
    )
    return float(sum(partials))


class kmeans_parallel():
    """K-means con inizializzazione parallela (k-means||) su Dask."""

    # --------------------------------------------------------------------

    def __init__(self, k, l, r=None):
        """
        Parameters
        ----------
        k : int
            Numero di cluster finali desiderati.
        l : int
            Fattore di oversampling: numero atteso di candidati campionati
            ad ogni round dell'inizializzazione parallela.
        r : int, optional
            Numero di round dell'inizializzazione parallela. Se None, viene
            risolto da ``resolve_rounds`` (regola del paper: 15 se
            l/k <= 0.1, altrimenti alpha * log(psi)).
        """
        self.k = k
        self.l = l  # oversampling factor
        self.r = r  # numero di round richiesto (None = auto)
        self.centroids = []
        self.starting_centroids = None  # impostato da compute_starting_centroids
        self.final_centroids = None     # impostato da fit
        self.n_iter_ = None    # iterazioni di Lloyd's eseguite da fit()
        self.n_rounds_ = None  # round k-means|| effettivamente eseguiti
        self.sampling_ = None  # schema di campionamento usato dal seeding

    # --------------------------------------------------------------------

    def compute_starting_centroids(self, X, alpha=1, l=None, max_iter=None, seed=None, track_centroids=False, policy="auto", sampling="bernoulli"):
        """Inizializzazione k-means|| parallela: seleziona un pool di
        candidati centroidi campionando iterativamente da X con
        probabilita' proporzionale alla distanza al quadrato dal centroide
        piu' vicino gia' scelto, poi li riduce a k centroidi finali con un
        k-means pesato (scikit-learn).

        Il numero di round e' risolto da ``resolve_rounds`` (policy="auto",
        regola del paper) oppure preso com'e' con policy="fixed"; il valore
        effettivamente eseguito resta in ``self.n_rounds_``.

        Con policy="fixed" e r=0: modalita' RANDOM BASELINE — k centri
        uniformi senza round ne' reclustering (punto r=0 dell'asse in
        Fig 5.2 del paper; e' anche il baseline Random di Table 3).

        sampling:
          - "bernoulli" (default): Algorithm 2 del paper — ogni punto
            campionato con probabilita' min(1, l*d2/cost);
          - "exact": ESATTAMENTE l punti per round senza reimmissione,
            probabilita' proporzionale a d^2 (protocollo della sola
            Fig 5.1; schema Efraimidis-Spirakis distribuito).

        Se track_centroids=True, salva in self.n_centroids_history_ il
        numero cumulativo di candidati centroidi dopo ogni round eseguito,
        utile per ispezionare la crescita del pool di candidati.

        Deterministico dato ``seed``: le estrazioni usano SeedSequence(seed)
        e un RNG per (partizione, round), non il RNG globale di NumPy.
        """
        if sampling not in ("bernoulli", "exact"):
            raise ValueError("sampling deve essere 'bernoulli' o 'exact'")
        if l is None:
            l = self.l
        self.sampling_ = sampling

        parts = _persist_matrices(X)
        shapes = dask.compute(*[dask.delayed(_matrix_shape, pure=True)(p) for p in parts])
        n_points = int(sum(s[0] for s in shapes))

        if track_centroids:
            self.n_centroids_history_ = []

        # SeedSequence padre: entropia deterministica se seed e' fornito,
        # casuale altrimenti. Da qui derivano TUTTE le estrazioni, su tre
        # rami figli INDIPENDENTI (spawn): centroide iniziale / indici
        # random-baseline, round di campionamento, reclustering pesato.
        # Usare la stessa sequenza per piu' Generator duplicherebbe lo
        # stesso stream (estrazioni correlate): ogni consumo ha il suo ramo.
        ss_init, ss_body, ss_reclust = np.random.SeedSequence(seed).spawn(3)

        # STEP 0: con policy="fixed" il numero di round non dipende dai dati
        # (psi non serve): si puo' cortocircuitare PRIMA di toccare X.
        # r=0 => RANDOM BASELINE: k indici uniformi senza reimmissione.
        r_request = self.r if max_iter is None else max_iter
        n_rounds_fixed = (
            resolve_rounds(l=l, k=self.k, r=r_request, alpha=alpha, psi=None, policy="fixed")
            if policy == "fixed" else None
        )
        if n_rounds_fixed == 0:
            rng0 = np.random.default_rng(ss_init)
            global_idx = np.sort(rng0.choice(n_points, size=self.k, replace=False))
            offsets = np.cumsum([0] + [int(s[0]) for s in shapes])
            fetch_tasks = []
            for j, p in enumerate(parts):
                loc = global_idx[(global_idx >= offsets[j]) & (global_idx < offsets[j + 1])]
                fetch_tasks.append(dask.delayed(_rows_at)(p, loc - offsets[j]))
            C = np.vstack(dask.compute(*fetch_tasks))
            self.centroids = [C]
            self.starting_centroids = C
            self.n_rounds_ = 0
            return

        # STEP 1: centroide iniziale casuale (uniforme su tutti i punti,
        # via indice estratto dal RNG dedicato)
        rng = np.random.default_rng(ss_init)
        initial_idx = int(rng.integers(n_points))
        offset = 0
        for p_idx, (m, _) in enumerate(shapes):
            if initial_idx < offset + m:
                local = initial_idx - offset
                initial_centroid = np.asarray(
                    dask.delayed(_row_at, pure=True)(
                        parts[p_idx], local
                    ).compute(),
                    dtype=np.float64,
                )
                break
            offset += m
        initial_centroid = initial_centroid.reshape(1, -1)
        self.centroids.append(initial_centroid)

        # stato per partizione: (m, 2) -> (d^2 minima, indice centroide).
        # Lo stato resta lato worker per TUTTO il seeding (catena di task
        # ritardati): al client arrivano solo gli scalari di costo, uno per
        # partizione per round — mai la matrice (m, 2) completa.
        state_delays = [
            dask.delayed(_init_state, pure=False)(p, initial_centroid[0])
            for p in parts
        ]

        # STEP 2: costo iniziale (solo scalari verso il client)
        psi = float(sum(dask.compute(*[
            dask.delayed(_state_cost, pure=False)(s) for s in state_delays
        ])))
        if psi == 0.0:
            # Tutti i punti coincidono con il centroide iniziale: non c'e'
            # nulla da campionare. Si rispetta il contratto della classe
            # (starting_centroids ha esattamente k righe) ripetendo il
            # centroide; il costo e' comunque 0.
            self.n_rounds_ = 0
            self.starting_centroids = np.repeat(initial_centroid, self.k, axis=0)
            return

        # STEP 3: numero di round, risolto in un unico punto (resolve_rounds)
        if policy == "fixed":
            n_rounds = n_rounds_fixed  # gia' noto dallo STEP 0 (> 0 qui)
        else:
            n_rounds = resolve_rounds(
                l=l, k=self.k, r=r_request, alpha=alpha, psi=psi, policy="auto"
            )

        # chiavi RNG figlie, una per partizione: indipendenti per costruzione.
        # I seed per (partizione, round) sono pre-derivati in blocco con
        # spawn (deterministico): nessun stream condiviso, nessun trucco
        # sull'entropia.
        child_seeds = ss_body.spawn(len(parts))
        round_seeds = [child.spawn(n_rounds) for child in child_seeds]

        cost = psi
        rounds_run = 0
        for round_idx in range(n_rounds):
            if cost == 0.0:
                break
            rounds_run += 1

            if sampling == "exact":
                # top-l LOCALI per partizione (chiavi Efraimidis-Spirakis):
                # solo l coppie (chiave, indice) per partizione attraversano
                # la rete, i punti restano sui worker.
                key_tasks = [
                    dask.delayed(_sample_round_exact, pure=False)(
                        p, s, l, round_seeds[j][round_idx]
                    )
                    for j, (p, s) in enumerate(zip(parts, state_delays))
                ]
                key_parts = dask.compute(*key_tasks)
                all_keys = np.concatenate([kp[0] for kp in key_parts])
                part_ids = np.concatenate(
                    [np.full(len(kp[0]), j, dtype=np.int64) for j, kp in enumerate(key_parts)]
                )
                local_ids = np.concatenate([kp[1] for kp in key_parts])

                # merge globale -> top-l (o meno se l'intero dataset ha
                # meno punti a peso positivo di l)
                order = np.argsort(all_keys)[::-1][:l]
                sel_part = part_ids[order]
                sel_loc = local_ids[order]

                # recupero delle sole righe scelte, una task per partizione
                fetch_tasks = []
                for j, p in enumerate(parts):
                    loc = np.sort(sel_loc[sel_part == j])
                    fetch_tasks.append(dask.delayed(_rows_at)(p, loc))
                fetched = dask.compute(*fetch_tasks)
                sampled = [f for f in fetched if f.shape[0] > 0]
            else:
                # probabilità di campionamento per ogni punto (vettore per
                # partizione), Algorithm 2 del paper
                sample_tasks = [
                    dask.delayed(_sample_round, pure=False)(
                        p, s, l, cost, round_seeds[j][round_idx]
                    )
                    for j, (p, s) in enumerate(zip(parts, state_delays))
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
                    for p, s in zip(parts, state_delays)
                ]
                # lo stato aggiornato NON viene raccolto sul client: ne
                # teniamo i riferimenti simbolici (input del round dopo) e
                # calcoliamo solo gli scalari di costo fusi in _update_state.
                state_delays = [u[0] for u in update_tasks]
                cost = float(sum(dask.compute(*[u[1] for u in update_tasks])))

            if track_centroids:
                self.n_centroids_history_.append(len(self.centroids))

        # round effettivamente eseguiti (puo' essere < n_rounds se il costo
        # e' arrivato a zero in anticipo): e' questo il valore che i driver
        # registrano nei risultati (colonna r_effective).
        self.n_rounds_ = rounds_run

        # STEP 7: pesi = numero di punti assegnati a ciascun candidato centroide
        # (k-vettori per partizione: unica riduzione finale sugli stati)
        weights = sum(
            dask.compute(*[dask.delayed(_partition_bincount, pure=True)(s, len(self.centroids))
                           for s in state_delays])
        )
        centroids_weights = weights.astype(np.float64)

        # STEP 8: riduzione finale a k centroidi con k-means pesato (scikit-learn)
        # n_init=1: il paper (Bahmani et al.) usa una singola inizializzazione
        # k-means++ per il reclustering, non i 10 restart di default di sklearn.
        # random_state deriva dal ramo ss_reclust della SeedSequence: senza,
        # il k-means++ interno di sklearn pescherebbe dal RNG globale e il
        # risultato non sarebbe riproducibile nemmeno a parita' di seed.
        reclustering_random_state = int(np.random.default_rng(ss_reclust).integers(2**31 - 1))
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
        nello stesso passaggio: nessun giro extra sui dati). Le etichette
        (m,) di ogni partizione NON viaggiano verso il client fra un'
        iterazione e l'altra: restano nel grafo Dask come input
        dell'iterazione successiva; al client arrivano solo somme (k,d),
        conteggi, costo e numero di cambiamenti. Se
        track_convergence=True, le quantita' vengono anche registrate in
        self.cost_history_ (inertia ad ogni iterazione) e self.iter_times_
        (tempo per iterazione).

        Al termine, self.n_iter_ contiene il numero di iterazioni di
        Lloyd's effettivamente eseguite (aggiornamenti dei centroidi
        completati).
        """
        if self.starting_centroids is None:
            raise RuntimeError(
                "fit(): chiamare prima compute_starting_centroids(X, ...)"
            )

        parts = _persist_matrices(X)
        n_partitions = len(parts)

        centroids_arr = np.vstack(self.starting_centroids)
        k = centroids_arr.shape[0]

        if track_convergence:
            self.cost_history_ = []
            self.iter_times_ = []

        # etichette dell'iterazione precedente, UNA referenza simbolica per
        # partizione (None alla prima iterazione). Sono Delayed nel grafo:
        # i vettori (m,) non lasciano mai i worker.
        prev_labels = [None] * n_partitions
        empty_warned = False

        for iteration in range(max_iter):
            iter_start = time.time()

            tasks = [
                dask.delayed(_lloyd_pass, pure=False)(p, prev_labels[j], centroids_arr)
                for j, p in enumerate(parts)
            ]
            # calcoliamo SOLO le riduzioni piccole t[:4]; l'elemento t[4]
            # (etichette della partizione) resta nel grafo come stato
            # distribuito per l'iterazione successiva.
            reductions = [t[:4] for t in tasks]
            results = dask.compute(*reductions)

            sums = np.zeros_like(centroids_arr)
            counts = np.zeros(k, dtype=np.int64)
            iter_cost = 0.0
            changed = 0
            for p_sums, p_counts, p_cost, p_changed in results:
                sums += p_sums
                counts += p_counts
                iter_cost += p_cost
                changed += p_changed

            # le nuove etichette diventano lo stato distribuito del giro dopo
            prev_labels = [t[4] for t in tasks]

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
        if self.final_centroids is None:
            raise RuntimeError(
                "classify(): chiamare prima fit() (o impostare final_centroids)"
            )
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
        if self.final_centroids is None:
            raise RuntimeError(
                "inertia(): chiamare prima fit() (o impostare final_centroids)"
            )
        return inertia_of_bag(X, self.final_centroids)
