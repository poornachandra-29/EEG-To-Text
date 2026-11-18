"""
agglomerative_model.py — Agglomerative Hierarchical Clustering (Ward)
=====================================================================
Mathematical Formulation:
    Ward linkage criterion minimizes total within-cluster variance increase:
        Delta_Var(A, B) = [ (n_A * n_B) / (n_A + n_B) ] * ||mu_A - mu_B||^2
    where n_A, n_B are cluster sizes and mu_A, mu_B are cluster mean vectors.

Faculty Explanation:
    Agglomerative clustering is a deterministic, bottom-up hierarchical method.
    Starting with each trial as an individual singleton cluster, it recursively
    merges the pair that minimizes the sum of squared differences.
    Ward linkage prevents the elongated 'chaining' artifacts common in single linkage,
    producing balanced and robust cognitive state hierarchies.
    For held-out test evaluation, out-of-sample trials are mapped via nearest
    centroid assignment in the standardized feature space.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def build_agglomerative_model(n_clusters: int = 5) -> AgglomerativeClustering:
    """Builds and returns an Agglomerative Clustering estimator with Ward linkage."""
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )


def get_agglomerative_centroids(labels: np.ndarray, X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Computes empirical mean centroids for each of the K agglomerated clusters."""
    centroids = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        mask = (labels == k)
        if np.any(mask):
            centroids[k] = X[mask].mean(axis=0)
    return centroids


def predict_agglomerative(centroids: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Assigns unseen test samples to the nearest cluster centroid in standardized space."""
    dists = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)
