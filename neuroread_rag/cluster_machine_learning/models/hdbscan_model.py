"""
hdbscan_model.py — Hierarchical Density-Based Spatial Clustering
================================================================
Mathematical Formulation:
    Mutual reachability distance:
        d_mreach-k(a, b) = max(core_k(a), core_k(b), d(a, b))
    Builds a minimum spanning tree, transforms it into a cluster tree hierarchy,
    and extracts the most persistent, stable clusters by maximizing excess of mass.

Faculty Explanation:
    HDBSCAN is a non-parametric, density-based clustering algorithm. Unlike
    centroid-based models that force every data point into one of K groups,
    HDBSCAN isolates atypical physiological artifacts (e.g. eye-tracker blinks,
    swallowing muscle artifacts) as unclassified outlier noise (label = -1).
    This ensures that discovered reading states are composed solely of dense,
    statistically coherent cognitive responses.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import HDBSCAN


def build_hdbscan_model(min_cluster_size: int = 50, min_samples: int = 10) -> HDBSCAN:
    """Builds and returns an HDBSCAN estimator storing cluster centroids."""
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        store_centers="centroid"
    )


def get_hdbscan_centroids(model: HDBSCAN, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Extracts non-noise cluster centroids."""
    if hasattr(model, "centroids_") and model.centroids_ is not None and len(model.centroids_) > 0:
        return model.centroids_
    unique_k = sorted([lbl for lbl in np.unique(labels) if lbl >= 0])
    centroids = np.zeros((len(unique_k), X.shape[1]))
    for idx, k in enumerate(unique_k):
        centroids[idx] = X[labels == k].mean(axis=0)
    return centroids


def predict_hdbscan(model: HDBSCAN, centroids: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Assigns unseen test samples to the nearest non-noise cluster centroid."""
    dists = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)
