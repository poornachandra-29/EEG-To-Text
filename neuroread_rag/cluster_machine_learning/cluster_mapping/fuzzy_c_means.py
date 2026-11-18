"""
fuzzy_c_means.py — Scikit-Learn Compatible Fuzzy C-Means Estimator
==================================================================
Implements standard Bezdek Fuzzy C-Means (FCM) clustering:
    - Objective: min J_m = sum_i sum_j (u_ij^m * ||x_i - c_j||^2)
    - Supports fit, fit_predict, predict, and predict_proba.
"""

import numpy as np


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering with soft membership degrees u_ij in [0, 1].
    """
    def __init__(self, n_clusters: int = 5, m: float = 1.3, max_iter: int = 150, tol: float = 1e-5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.u_ = None

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        # Initialize centers with randomly chosen distinct samples
        idx = rng.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy().astype(float)

        power = 2.0 / (self.m - 1.0)
        for it in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            # Calculate distances to cluster centers
            dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
            dists = np.fmax(dists, 1e-10)

            # Compute memberships
            inv_dists = dists ** (-power)
            u = inv_dists / inv_dists.sum(axis=1, keepdims=True)

            # Update centers
            um = u ** self.m
            self.cluster_centers_ = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-10)

            if np.linalg.norm(self.cluster_centers_ - old_centers) < self.tol:
                break

        self.u_ = u
        self.labels_ = np.argmax(u, axis=1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        dists = np.fmax(dists, 1e-10)
        power = 2.0 / (self.m - 1.0)
        inv_dists = dists ** (-power)
        return inv_dists / inv_dists.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
