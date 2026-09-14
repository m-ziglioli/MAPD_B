"""
kmeans_parallel.py
==================
Implementation of the k-means|| initialization algorithm (parallel
k-means++, Bahmani et al., VLDB 2012) on Dask, followed by distributed
Lloyd's iterations (fit).

Computational engine
--------------------
All point-level operations are vectorized PER PARTITION: the distributed
dataset (a ``dask.array`` of 2-D chunks, or a legacy ``dask.bag`` of rows)
is decomposed into delayed partitions, each stacked into a dense ``(m, d)``
matrix, and every pass of the algorithm becomes one task per partition
returning only the required reductions (per-cluster sums, counts, cost,
changed-label count). This eliminates
millions of Python calls and any row shuffle: only k x d matrices cross
partition boundaries.

Seeding is deterministic: every draw (initial centroid, per-round Bernoulli
sampling, weighted-reclustering random_state) derives from
``SeedSequence(seed)``, with one RNG per (partition, round) — the scheduler
execution order does not influence the result.

Typical usage (from a notebook):

    from src.kmeans_parallel import kmeans_parallel

    clf = kmeans_parallel(k=500, l=200, r=5)
    clf.compute_starting_centroids(X, seed=42)
    clf.fit(X, max_iter=10)
    labels = clf.classify(X)
"""

import time
import warnings

import dask
import dask.array as da
import dask.bag as db
from sklearn.cluster import KMeans
import numpy as np


# ----------------------------------------------------------------------------
# Vectorized per-partition helpers 
# ----------------------------------------------------------------------------

def _stack_rows(rows):
    """List of 1-D rows -> dense (m, d) float64 matrix."""
    arrs = [np.asarray(r, dtype=np.float64) for r in rows]
    if not arrs:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(arrs)


def _bag_to_matrices(X):
    """dask.array or dask.bag -> flat list of Delayed, one per partition,
    each a dense (m, d) matrix.

    A dask.array already has 2-D chunks; note that ``Array.to_delayed()``
    returns a nested ndarray on the chunk grid (unlike bag/dataframe, which
    return a flat list), so it must be flattened. A bag of rows is stacked with
    _stack_rows."""
    if isinstance(X, da.Array):
        return list(X.to_delayed().ravel().tolist())
    return [dask.delayed(_stack_rows, pure=True)(p) for p in X.to_delayed()]


def _persist_matrices(X):
    """Like _bag_to_matrices, but the per-partition matrices are
    MATERIALIZED once (futures on the cluster, cache in the local
    scheduler). Every subsequent seeding/fit task references these nodes:
    without persistence each round/iteration would re-stack the partitions
    from scratch, multiplying the most expensive work of the engine."""
    parts = _bag_to_matrices(X)
    if parts:
        parts = list(dask.persist(*parts))
    return parts


def _matrix_shape(M):
    return M.shape


def _row_at(M, i):
    """i-th row of a partition matrix (independent copy)."""
    return M[i].copy()


def _state_cost(state):
    """Sum of the current minimum d^2 over one partition (scalar)."""
    return float(state[:, 0].sum())


def _pairwise_d2(M, C):
    """d2[i, j] = ||M[i] - C[j]||^2 via the quadratic expansion
    (BLAS-friendly).

    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c : the distance between all rows
    and all centroids is a single matmul. Clipping at 0 avoids small
    negative values from floating-point arithmetic.
    """
    m_sq = np.einsum("ij,ij->i", M, M)
    # ->i: one norm per CENTROID (row of C). With ->j one would get the
    # per-column sums (d values) and the broadcasting with (M @ C.T) (m, t)
    # would fail or, if t == d, silently produce wrong distances.
    c_sq = np.einsum("ij,ij->i", C, C)
    d2 = m_sq[:, None] + c_sq[None, :] - 2.0 * (M @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _pairwise_d2_argmin_chunked(M, C, chunk_k=100):
    """For every point in M, squared distance to the nearest centroid in C
    — WITHOUT ever building the full (m, k) distance matrix.

    Why: the one-shot version (_pairwise_d2) materializes an
    (n_points, n_centroids) matrix per partition. With large k (500-1000)
    and large partitions this can weigh several MB in memory (e.g. 8
    partitions/worker, k=1000, 4M points total: ~1e8 elements per partition
    pass = 800 MB each.

    How: centroids are visited in groups ("chunks") of chunk_k at a time.
    For each group we compute distances only towards that group and keep
    track of the best result seen so far (minimum distance and
    corresponding centroid index). The result is identical, but at most
    (m, chunk_k) values are alive at any moment.

    Parameters
    ----------
    M : array (m, d)
        Points of this partition.
    C : array (k, d)
        All current centroids.
    chunk_k : int
        How many centroids to consider at a time. Smaller = less memory,
        more loop iterations; larger = more memory, fewer iterations.

    Returns
    -------
    best_dist : array (m,)
        Squared distance from the nearest centroid, per point.
    best_idx : array (m,) of int
        Index (0-based) of the nearest centroid, per point.
    """
    m = M.shape[0]
    if m == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)

    # Best result found so far per point: start with "infinity" so the
    # first chunk always wins, and index 0 as a placeholder.
    best_dist = np.full(m, np.inf)
    best_idx = np.zeros(m, dtype=np.int64)

    # ||x||^2 per point, computed once outside the loop (depends only on M).
    m_sq = np.einsum("ij,ij->i", M, M)

    for start in range(0, C.shape[0], chunk_k):
        end = start + chunk_k
        C_chunk = C[start:end]  # shape: (up to chunk_k, d)

        c_sq = np.einsum("ij,ij->i", C_chunk, C_chunk)

        # Squared distances between every point of M and every centroid of
        # this chunk: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x . c).
        # Shape (m, chunk_k) — much smaller than the full (m, k) matrix.
        d2_chunk = m_sq[:, None] + c_sq[None, :] - 2.0 * (M @ C_chunk.T)
        # Small float rounding errors can give slightly negative values.
        np.maximum(d2_chunk, 0.0, out=d2_chunk)

        # Best centroid WITHIN THIS CHUNK per point.
        local_best_idx = d2_chunk.argmin(axis=1)                # local index (0..chunk_k-1)
        local_best_dist = d2_chunk[np.arange(m), local_best_idx]

        # Update only the points for which this chunk found something closer.
        is_better = local_best_dist < best_dist
        best_dist[is_better] = local_best_dist[is_better]
        # local_best_idx is relative to the chunk (starts at 0): add
        # "start" to obtain the true index in the full centroid vector.
        best_idx[is_better] = local_best_idx[is_better] + start

    return best_dist, best_idx


def _lloyd_pass(M, prev_labels, C):
    """One Lloyd's assignment iteration on one partition.

    Returns (sums (k,d), counts (k,), cost, changed, labels (m,) int32).
    ``changed`` compares the labels with those of the previous iteration
    (fused into the pass: no extra sweep over the data for the strict
    convergence criterion).
    """
    k = C.shape[0]
    if M.shape[0] == 0:
        return (np.zeros((k, C.shape[1])), np.zeros(k, dtype=np.int64),
                0.0, 0, np.empty(0, dtype=np.int32))

    # Chunked version: never builds the full (m, k) distance matrix, which
    # can occupy several GB and cause out-of-memory.
    best_dist, labels = _pairwise_d2_argmin_chunked(M, C)
    sums = np.zeros((k, M.shape[1]))
    np.add.at(sums, labels, M)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    # best_dist already holds the distance to the assigned centroid for
    # every point: sum it directly.
    cost = float(best_dist.sum())

    if prev_labels is None:
        changed = int(M.shape[0])
    else:
        changed = int((labels != prev_labels).sum())
    return sums, counts, cost, changed, labels.astype(np.int32)


def _init_state(M, c0):
    """Initial k-means|| state per partition: column 0 = d^2(x, c0),
    column 1 = index of the nearest centroid (0)."""
    if M.shape[0] == 0:
        return np.empty((0, 2))
    d2 = _pairwise_d2(M, c0.reshape(1, -1))[:, 0]  # single centroid: no chunking needed
    return np.column_stack([d2, np.zeros(len(M))])


def _sample_round(M, state, l, cost, round_seed_seq):
    """Vectorized Bernoulli sampling on one partition.

    Each point is sampled with probability min(1, l * d2 / cost) using a
    local RNG derived from the child SeedSequence of (partition, round),
    pre-derived by the caller: deterministic and parallel-safe, with no
    stream sharing between draws.
    Returns the (t, d) matrix of sampled points.
    """
    if M.shape[0] == 0 or cost <= 0.0:
        return np.empty((0, M.shape[1] if M.ndim == 2 else 0))
    rng = np.random.default_rng(round_seed_seq)
    probs = np.minimum(1.0, state[:, 0] * l / cost)
    mask = rng.random(M.shape[0]) < probs
    return M[mask]


def _sample_round_exact(M, state, l, round_seed_seq):
    """EXACT sampling of l points per round, without replacement, with
    probability proportional to d^2(x, C) — the protocol of Fig 5.1 of the
    paper (Bahmani et al. use this variant ONLY for Fig 5.1).

    Efraimidis-Spirakis scheme (weighted sampling without replacement):
    key u^(1/w) with u~U(0,1), w = d^2; the LOCAL top-l of each partition,
    once merged, contains the GLOBAL top-l (property of the maximum: top-l
    of a union = union of the top-l). Thus only l (key, index) pairs per
    partition cross the client/cluster boundary, never the points.

    Zero-distance points have zero weight and are never sampleable
    (consistent with Bernoulli, where p = min(1, l*d2/cost) = 0).

    Returns (keys (t,), local_indices (t,) int64): the keys feed the global
    merge on the client, the indices retrieve the chosen rows.
    """
    if M.shape[0] == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(round_seed_seq)
    u = rng.random(M.shape[0])
    w = state[:, 0].astype(np.float64)
    nz = w > 0.0
    # key -inf => point not sampleable (zero weight). Power form (not
    # exp/log): exact u=0 raises no warning and gives key 0.
    keys = np.full(M.shape[0], -np.inf)
    keys[nz] = u[nz] ** (1.0 / w[nz])
    l_loc = int(min(l, nz.sum()))
    if l_loc <= 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    top = np.argsort(keys)[-l_loc:]
    return keys[top], top.astype(np.int64)


def _rows_at(M, idx):
    """Rows of a partition at local indices ``idx`` (fancy indexing); used
    both by the r=0 path and by exact-sample retrieval."""
    if len(idx) == 0:
        return np.empty((0, M.shape[1] if M.ndim == 2 else 0))
    return M[idx]


def _update_state(M, state, new_centroids, start_idx):
    """Update the (min d^2, centroid index) state after adding the
    candidates ``new_centroids`` (t, d) starting at index start_idx.

    Returns (updated state, partial sum of min d^2): the round cost is
    fused into the update, so only the scalar crosses the client/cluster
    boundary, not the state matrix.
    """
    if M.shape[0] == 0 or new_centroids.shape[0] == 0:
        return state, float(state[:, 0].sum())
    best_dist, best_idx = _pairwise_d2_argmin_chunked(M, new_centroids)
    closer = best_dist < state[:, 0]
    out = state.copy()
    out[closer, 0] = best_dist[closer]
    out[closer, 1] = (best_idx[closer] + start_idx).astype(state[:, 1].dtype)
    return out, float(out[:, 0].sum())


def _partition_bincount(state, n_centers):
    """Histogram of assignments (reclustering weights) per partition."""
    if state.shape[0] == 0:
        return np.zeros(n_centers, dtype=np.int64)
    return np.bincount(state[:, 1].astype(np.int64), minlength=n_centers)


def resolve_rounds(l, k, r=None, alpha=1.0, psi=None, policy="auto"):
    """Number of k-means|| rounds to run, resolved in ONE place.

    policy="auto" (default) follows the paper protocol (Bahmani et al.,
    VLDB 2012):
      - if l/k <= 0.1 more rounds are needed to accumulate at least k
        candidates: 15 rounds are used, regardless of ``r``;
      - otherwise an explicit ``r`` wins;
      - without ``r`` the estimate is round(alpha * log(psi)).

    policy="fixed" always uses ``r`` exclusively (mandatory): an explicit
    escape hatch to bypass the paper rule. With r=0 the caller
    (compute_starting_centroids) gets the "random baseline" mode: k uniform
    centers with no rounds and no reclustering.

    Note: the l/k<=0.1 -> 15 rule is intentionally kept even when ``r`` is
    provided (historical project behavior); drivers record the EFFECTIVE
    number of rounds (``n_rounds_`` / ``r_effective`` CSV column) so sweeps
    compare configurations on the value actually executed.
    """
    if policy not in ("auto", "fixed"):
        raise ValueError("policy must be 'auto' or 'fixed'")
    if policy == "fixed":
        if r is None:
            raise ValueError("policy='fixed' requires an explicit number of rounds r")
        if int(r) < 0:
            raise ValueError("r cannot be negative")
        return int(r)
    # r=0 ALWAYS means random baseline
    if r == 0:
        return 0
    # policy == "auto"
    if l / k <= 0.1:
        return 15
    if r is not None:
        if int(r) < 0:
            raise ValueError("r cannot be negative")
        return int(r)
    if psi is None or psi <= 0:
        raise ValueError("without an explicit r, psi > 0 is needed to estimate the rounds")
    return max(1, int(round(alpha * float(np.log(psi)))))


def _labels_partition(M, C):
    """Nearest-centroid labels of one partition, as a list of ints (Bag
    semantics for classify())."""
    if M.shape[0] == 0:
        return []
    _, best_idx = _pairwise_d2_argmin_chunked(M, C)
    return best_idx.tolist()


def _inertia_partial(M, C):
    """Partial sum of d^2 to the nearest centroid, per partition."""
    if M.shape[0] == 0:
        return 0.0
    best_dist, _ = _pairwise_d2_argmin_chunked(M, C)
    return float(best_dist.sum())


def inertia_of_bag(X, centroids):
    """Inertia (sum of d^2 to the nearest centroid) over the whole
    distributed dataset, with one vectorized task per partition. Single
    shared computation point used by kmeans_parallel.inertia() and
    benchmark.calculate_inertia()."""
    centroids_arr = np.vstack(centroids)
    partials = dask.compute(
        *[dask.delayed(_inertia_partial, pure=False)(p, centroids_arr)
          for p in _bag_to_matrices(X)]
    )
    return float(sum(partials))


class kmeans_parallel():
    """K-means with parallel initialization (k-means||) on Dask."""

    # --------------------------------------------------------------------

    def __init__(self, k, l, r=None):
        """
        Parameters
        ----------
        k : int
            Number of final clusters desired.
        l : int
            Oversampling factor: expected number of candidates sampled at
            each round of the parallel initialization.
        r : int, optional
            Number of rounds of the parallel initialization. If None, it is
            resolved by ``resolve_rounds`` (paper rule: 15 if l/k <= 0.1,
            otherwise alpha * log(psi)).
        """
        self.k = k
        self.l = l  # oversampling factor
        self.r = r  # requested number of rounds (None = auto)
        self.centroids = []
        self.starting_centroids = None  # set by compute_starting_centroids
        self.final_centroids = None     # set by fit
        self.n_iter_ = None    # Lloyd's iterations executed by fit()
        self.n_rounds_ = None  # k-means|| rounds actually executed
        self.sampling_ = None  # sampling scheme used by the seeding

    # --------------------------------------------------------------------

    def compute_starting_centroids(self, X, alpha=1, l=None, max_iter=None, seed=None, track_centroids=False, policy="auto", sampling="bernoulli"):
        """Parallel k-means|| initialization: selects a pool of candidate
        centroids by iteratively sampling from X with probability
        proportional to the squared distance from the nearest already-chosen
        centroid, then reduces them to k final centroids with a weighted
        k-means (scikit-learn).

        The number of rounds is resolved by ``resolve_rounds``
        (policy="auto", paper rule) or taken as-is with policy="fixed"; the
        value actually executed is stored in ``self.n_rounds_``.

        With policy="fixed" and r=0: RANDOM BASELINE mode — k uniform
        centers with no rounds and no reclustering (the r=0 point of the
        axis in Fig 5.2 of the paper; also the Random baseline of Table 3).

        sampling:
          - "bernoulli" (default): Algorithm 2 of the paper — each point
            sampled with probability min(1, l*d2/cost);
          - "exact": EXACTLY l points per round without replacement,
            probability proportional to d^2 (protocol of Fig 5.1 only;
            distributed Efraimidis-Spirakis scheme).

        If track_centroids=True, stores in self.n_centroids_history_ the
        cumulative number of candidate centroids after each executed round,
        useful to inspect the growth of the candidate pool.

        Deterministic given ``seed``: draws use SeedSequence(seed) and one
        RNG per (partition, round), not NumPy's global RNG.
        """
        if sampling not in ("bernoulli", "exact"):
            raise ValueError("sampling must be 'bernoulli' or 'exact'")
        if l is None:
            l = self.l
        self.sampling_ = sampling

        parts = _persist_matrices(X)
        shapes = dask.compute(*[dask.delayed(_matrix_shape, pure=True)(p) for p in parts])
        n_points = int(sum(s[0] for s in shapes))

        if track_centroids:
            self.n_centroids_history_ = []

        # Parent SeedSequence: deterministic entropy if seed is given,
        # random otherwise. ALL draws derive from here, on three
        # INDEPENDENT child branches (spawn): initial centroid /
        # random-baseline indices, sampling rounds, weighted reclustering.
        # Reusing the same sequence for multiple Generators would duplicate
        # the same stream (correlated draws): each consumption gets its own
        # branch.
        ss_init, ss_body, ss_reclust = np.random.SeedSequence(seed).spawn(3)

        # STEP 0: with policy="fixed" the number of rounds does not depend
        # on the data (psi is not needed): it can be short-circuited BEFORE
        # touching X. r=0 => RANDOM BASELINE: k uniform indices without
        # replacement.
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

        # STEP 1: uniform random initial centroid (over all points, via an
        # index drawn from the dedicated RNG)
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

        # Per-partition state: (m, 2) -> (min d^2, centroid index). The
        # state stays worker-side for the WHOLE seeding (chain of delayed
        # tasks): only cost scalars reach the client, one per partition per
        # round — never the full (m, 2) matrix.
        state_delays = [
            dask.delayed(_init_state, pure=False)(p, initial_centroid[0])
            for p in parts
        ]

        # STEP 2: initial cost (scalars only towards the client)
        psi = float(sum(dask.compute(*[
            dask.delayed(_state_cost, pure=False)(s) for s in state_delays
        ])))
        if psi == 0.0:
            # All points coincide with the initial centroid: nothing to
            # sample. The class contract (starting_centroids has exactly k
            # rows) is honored by repeating the centroid; the cost is 0
            # either way.
            self.n_rounds_ = 0
            self.starting_centroids = np.repeat(initial_centroid, self.k, axis=0)
            return

        # STEP 3: number of rounds, resolved in a single place (resolve_rounds)
        if policy == "fixed":
            n_rounds = n_rounds_fixed  # already known from STEP 0 (> 0 here)
        else:
            n_rounds = resolve_rounds(
                l=l, k=self.k, r=r_request, alpha=alpha, psi=psi, policy="auto"
            )

        # Child RNG keys, one per partition: independent by construction.
        # The (partition, round) seeds are pre-derived in bulk with spawn
        # (deterministic).
        child_seeds = ss_body.spawn(len(parts))
        round_seeds = [child.spawn(n_rounds) for child in child_seeds]

        cost = psi
        rounds_run = 0
        for round_idx in range(n_rounds):
            if cost == 0.0:
                break
            rounds_run += 1

            if sampling == "exact":
                # LOCAL top-l per partition (Efraimidis-Spirakis keys):
                # only l (key, index) pairs per partition cross the
                # network, the points stay on the workers.
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

                # Global merge -> top-l (or fewer if the whole dataset has
                # fewer positive-weight points than l)
                order = np.argsort(all_keys)[::-1][:l]
                sel_part = part_ids[order]
                sel_loc = local_ids[order]

                # Retrieve only the chosen rows, one task per partition
                fetch_tasks = []
                for j, p in enumerate(parts):
                    loc = np.sort(sel_loc[sel_part == j])
                    fetch_tasks.append(dask.delayed(_rows_at)(p, loc))
                fetched = dask.compute(*fetch_tasks)
                sampled = [f for f in fetched if f.shape[0] > 0]
            else:
                # Per-point sampling probability (vector per partition),
                # Algorithm 2 of the paper
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
                # The updated state is NOT collected on the client: we keep
                # symbolic references to it (input of the next round) and
                # compute only the cost scalars fused in _update_state.
                # Persisting the full result (state + cost) materializes the
                # state on the workers and keeps the task graph flat.
                persisted_results = list(dask.persist(*update_tasks))
                state_delays = [r[0] for r in persisted_results]
                cost = float(sum(dask.compute(*[r[1] for r in persisted_results])))

            if track_centroids:
                self.n_centroids_history_.append(len(self.centroids))

        # Rounds actually executed (can be < n_rounds if the cost reached
        # zero early): this is the value drivers record in the results
        # (r_effective column).
        self.n_rounds_ = rounds_run

        # STEP 7: weights = number of points assigned to each candidate
        # centroid (k-vectors per partition: the only final reduction over
        # the states)
        weights = sum(
            dask.compute(*[dask.delayed(_partition_bincount, pure=True)(s, len(self.centroids))
                           for s in state_delays])
        )
        centroids_weights = weights.astype(np.float64)

        # STEP 8: final reduction to k centroids with weighted k-means
        # (scikit-learn). n_init=1: the paper (Bahmani et al.) uses a
        # single k-means++ initialization for the reclustering, not
        # sklearn's 10 default restarts. random_state derives from the
        # ss_reclust branch of the SeedSequence: without it, sklearn's
        # internal k-means++ would draw from the global RNG and the result
        # would not be reproducible even at equal seed.
        reclustering_random_state = int(np.random.default_rng(ss_reclust).integers(2**31 - 1))
        kmeans = KMeans(n_clusters=self.k, n_init=1, random_state=reclustering_random_state)
        kmeans.fit(np.vstack(self.centroids), sample_weight=centroids_weights)
        self.starting_centroids = kmeans.cluster_centers_

    # --------------------------------------------------------------------

    def fit(self, X, max_iter=100, tol=1e-4, track_convergence=False):
        """Standard Lloyd's K-means, starting from the centroids computed
        by compute_starting_centroids, executed in a distributed way on X.

        Single code path: each iteration runs one task per partition that
        computes assignments, per-cluster sums, cost and number of changed
        labels (strict convergence criterion fused into the same pass: no
        extra sweep over the data). The (m,) labels of each partition do
        NOT travel to the client between iterations: they stay in the Dask
        graph as input of the next iteration; only (k,d) sums, counts, cost
        and the change count reach the client. If track_convergence=True,
        the quantities are also recorded in self.cost_history_ (inertia at
        each iteration) and self.iter_times_ (time per iteration).

        At the end, self.n_iter_ holds the number of Lloyd's iterations
        actually executed (completed centroid updates).
        """
        if self.starting_centroids is None:
            raise RuntimeError(
                "fit(): call compute_starting_centroids(X, ...) first"
            )

        parts = _persist_matrices(X)
        n_partitions = len(parts)

        centroids_arr = np.vstack(self.starting_centroids)
        k = centroids_arr.shape[0]

        if track_convergence:
            self.cost_history_ = []
            self.iter_times_ = []

        # Labels of the previous iteration, ONE symbolic reference per
        # partition (None at the first iteration). They are Delayed in the
        # graph: the (m,) vectors never leave the workers.
        prev_labels = [None] * n_partitions
        empty_warned = False

        for iteration in range(max_iter):
            iter_start = time.time()

            # Materializing the per-partition labels on the
            # workers (memory ~ n_points * 4 bytes) keeps the task graph
            # flat; recomputing them lazily each iteration would instead
            # make the graph grow without bound.
            tasks = [
                dask.delayed(_lloyd_pass)(p, prev_labels[j], centroids_arr)
                for j, p in enumerate(parts)
            ]
            tasks = list(dask.persist(*tasks))  # materialize now, cut graph growth
            # Compute ONLY the small reductions t[:4]; element t[4] (the
            # partition labels) stays in the graph as distributed state for
            # the next iteration.
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

            # The new labels become the distributed state of the next
            # iteration (references to the PERSISTED result, not a symbolic
            # chain).
            prev_labels = [t[4] for t in tasks]

            new_centroids = centroids_arr.copy()
            populated = counts > 0
            new_centroids[populated] = sums[populated] / counts[populated, None]
            if not empty_warned and not populated.all():
                warnings.warn(
                    f"{int((~populated).sum())} empty clusters after assignment: "
                    "the corresponding centroids keep their previous-iteration value"
                )
                empty_warned = True

            self.final_centroids = new_centroids
            # Number of completed Lloyd's updates
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

            if np.linalg.norm(new_centroids - centroids_arr) < tol * max(1, np.linalg.norm(centroids_arr)):
                print(f"Stopped at Lloyd's iteration {iteration} due to relative tolerance threshold {tol}")
                break

            centroids_arr = new_centroids

    # --------------------------------------------------------------------

    def classify(self, X):
        """Returns a Dask Bag with the index of the nearest cluster for
        every point of X, using the final centroids computed by fit()."""
        if self.final_centroids is None:
            raise RuntimeError(
                "classify(): call fit() first (or set final_centroids)"
            )
        centroids_arr = np.vstack(self.final_centroids)
        label_delays = [
            dask.delayed(_labels_partition, pure=False)(p, centroids_arr)
            for p in _bag_to_matrices(X)
        ]
        return db.from_delayed(label_delays)

    # --------------------------------------------------------------------

    def inertia(self, X):
        """Computes the inertia (sum of squared distances from the final
        centroids) over the whole dataset X."""
        if self.final_centroids is None:
            raise RuntimeError(
                "inertia(): call fit() first (or set final_centroids)"
            )
        return inertia_of_bag(X, self.final_centroids)
