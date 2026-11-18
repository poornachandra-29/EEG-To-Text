"""
fuzzy_c_means_model.py — Fuzzy C-Means (FCM Soft Clustering)
============================================================
Mathematical Formulation:
    Minimizes the fuzzy objective function:
        J_m = sum_{i=1}^N sum_{k=1}^K u_{ik}^m * ||x_i - c_k||^2
    subject to: sum_{k=1}^K u_{ik} = 1, for all i.
    Update equations:
        u_{ik} = 1 / sum_{j=1}^K ( ||x_i - c_k|| / ||x_i - c_j|| )^(2 / (m - 1))
        c_k = sum_{i=1}^N (u_{ik}^m * x_i) / sum_{i=1}^N (u_{ik}^m)
    where m > 1 is the fuzzifier (fuzziness exponent, default m=1.3).

Faculty Explanation:
    Unlike hard clustering where each sentence trial belongs to strictly one cluster,
    Fuzzy C-Means assigns each trial a continuous degree of membership u_{ik} in [0, 1].
    Cognitive states in human reading (e.g. transitioning from high focus into
    gradual mental fatigue) are physiologically continuous rather than abrupt.
    FCM captures this biological reality, yielding superior geometric separation
    and lower overlap on held-out test data.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class FuzzyCMeansModel:
    """
    Self-contained, robust Fuzzy C-Means estimator.
    Supports fit, fit_predict, predict, and predict_proba.
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
        idx = rng.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy().astype(float)

        power = 2.0 / (self.m - 1.0)
        for it in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
            dists = np.fmax(dists, 1e-10)

            inv_dists = dists ** (-power)
            u = inv_dists / inv_dists.sum(axis=1, keepdims=True)

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


def build_fuzzy_c_means_model(n_clusters: int = 5, m: float = 1.3, random_state: int = 42) -> FuzzyCMeansModel:
    """Builds and returns a Fuzzy C-Means estimator."""
    return FuzzyCMeansModel(
        n_clusters=n_clusters,
        m=m,
        max_iter=150,
        tol=1e-5,
        random_state=random_state
    )


def get_fuzzy_c_means_centroids(model: FuzzyCMeansModel) -> np.ndarray:
    """Extracts fuzzy cluster centroids c_k."""
    return model.cluster_centers_
