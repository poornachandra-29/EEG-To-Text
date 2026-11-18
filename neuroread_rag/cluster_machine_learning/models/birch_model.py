"""
birch_model.py — Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)
=====================================================================================
Mathematical Formulation:
    Clustering Feature (CF) summary entry:
        CF = (N, LS, SS)
        N  = Number of data points in subcluster
        LS = Linear sum of points (sum_{i=1}^N x_i)
        SS = Square sum of points (sum_{i=1}^N ||x_i||^2)
    Cluster radius: R = sqrt( (SS / N) - ||LS / N||^2 )
    Cluster centroid: X_0 = LS / N

Faculty Explanation:
    BIRCH is an incremental, tree-structured clustering algorithm designed for
    efficient, sub-linear memory scaling. By compressing raw trial data into
    compact Clustering Feature (CF) trees, BIRCH creates a hierarchical summary
    without repeatedly scanning the full dataset. It then applies a global
    partitioning step to produce exactly K=5 cognitive reading clusters.
    Importantly, BIRCH natively supports out-of-sample prediction (predict) on unseen
    test participants by routing new points through the leaf CF-tree.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import Birch


def build_birch_model(n_clusters: int = 5, threshold: float = 0.5, branching_factor: int = 50) -> Birch:
    """Builds and returns a BIRCH clustering estimator."""
    return Birch(
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor
    )


def get_birch_centroids(model: Birch) -> np.ndarray:
    """Extracts cluster centroids from the fitted BIRCH model."""
    if hasattr(model, "subcluster_centers_"):
        return model.subcluster_centers_[:model.n_clusters]
    return np.array([])
