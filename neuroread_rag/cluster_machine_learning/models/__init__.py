"""
models package — Modular Multi-Algorithm Clustering Suite for NeuroRead-RAG
============================================================================
Contains individual model definition files for:
    1. GMM                      -> gmm_model.py
    2. HDBSCAN                  -> hdbscan_model.py
    3. K-Means                  -> kmeans_model.py
    4. Fuzzy C-Means            -> fuzzy_c_means_model.py
    5. Agglomerative Clustering -> agglomerative_model.py
    6. BIRCH                    -> birch_model.py
"""

from .gmm_model import build_gmm_model, get_gmm_centroids
from .hdbscan_model import build_hdbscan_model, get_hdbscan_centroids, predict_hdbscan
from .kmeans_model import build_kmeans_model, get_kmeans_centroids
from .fuzzy_c_means_model import build_fuzzy_c_means_model, get_fuzzy_c_means_centroids, FuzzyCMeansModel
from .agglomerative_model import build_agglomerative_model, get_agglomerative_centroids, predict_agglomerative
from .birch_model import build_birch_model, get_birch_centroids

__all__ = [
    "build_gmm_model", "get_gmm_centroids",
    "build_hdbscan_model", "get_hdbscan_centroids", "predict_hdbscan",
    "build_kmeans_model", "get_kmeans_centroids",
    "build_fuzzy_c_means_model", "get_fuzzy_c_means_centroids", "FuzzyCMeansModel",
    "build_agglomerative_model", "get_agglomerative_centroids", "predict_agglomerative",
    "build_birch_model", "get_birch_centroids"
]
