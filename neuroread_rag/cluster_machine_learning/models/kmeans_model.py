"""
kmeans_model.py — K-Means Clustering (k-means++)
================================================
Mathematical Formulation:
    Minimizes within-cluster sum of squares (WCSS / Inertia):
        J = sum_{i=1}^N sum_{k=1}^K r_{ik} * ||x_i - mu_k||^2
    where r_{ik} in {0, 1} is a binary cluster assignment indicator
    and mu_k is the geometric center (mean) of cluster k.

Faculty Explanation:
    K-Means is the foundational centroid-based partitioning benchmark in machine learning.
    Using k-means++ seeding, initial centers are chosen with probability proportional
    to their squared distance from nearest existing centers, accelerating convergence
    and avoiding sub-optimal local minima. It provides an isotropic, hard-partitioning
    baseline against which soft and density-based models are contrasted.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans


def build_kmeans_model(n_clusters: int = 5, random_state: int = 42) -> KMeans:
    """Builds and returns a K-Means estimator with k-means++ initialization."""
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=random_state
    )


def get_kmeans_centroids(model: KMeans) -> np.ndarray:
    """Extracts cluster centroids mu_k."""
    return model.cluster_centers_
