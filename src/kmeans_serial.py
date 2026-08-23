"""
kmeans_serial.py
================
Serial baselines for comparison with k-means||: classic k-means++ seeding
(via scikit-learn) and uniform Random seeding, both followed by standard
Lloyd's iteration on a single machine.

Same two-phase API as kmeans_parallel (compute_starting_centroids -> fit),
so both can be driven the same way from a comparison script.
"""

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus


class kmeans_serial:
    """Serial k-means with either k-means++ or Random seeding."""

    def __init__(self, k, init="k-means++"):
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'")
        self.k = k
        self.init = init
        self.starting_centroids = None
        self.final_centroids = None
        self.n_iter_ = None

    def compute_starting_centroids(self, X, seed=None, n_local_trials=None):
        """Seed k centroids from X (no Lloyd's iterations yet).

        n_local_trials only applies to init="k-means++": None uses
        sklearn's default (greedy k-means++), 1 replicates the plain
        Algorithm 1 of Bahmani et al.
        """
        if self.init == "k-means++":
            centers, _ = kmeans_plusplus(
                X, n_clusters=self.k, random_state=seed, n_local_trials=n_local_trials
            )
            self.starting_centroids = centers
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=self.k, replace=False)
            self.starting_centroids = X[idx]

    def fit(self, X, max_iter=100, tol=1e-4):
        """Run Lloyd's iteration starting from self.starting_centroids."""
        km = KMeans(
            n_clusters=self.k,
            init=self.starting_centroids,
            n_init=1,
            max_iter=max_iter,
            tol=tol,
        )
        km.fit(X)
        self.final_centroids = km.cluster_centers_
        self.n_iter_ = km.n_iter_
