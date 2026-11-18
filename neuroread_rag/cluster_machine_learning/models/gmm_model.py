"""
gmm_model.py — Gaussian Mixture Model (Full Covariance)
=======================================================
Mathematical Formulation:
    p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
    where pi_k are mixture weights, mu_k are cluster mean vectors,
    and Sigma_k are full unconstrained covariance matrices.

Faculty Explanation:
    GMM is a probabilistic clustering framework that models data as a mixture
    of K multivariate Gaussian distributions. Full covariance modeling allows
    the algorithm to capture physiological interactions between correlated
    features (e.g. Total Reading Time strongly correlating with Gaze Duration,
    and Frontal Theta correlating with Alpha during cognitive fatigue).
    Crucially, GMM yields continuous posterior probabilities P(C_k | x),
    enabling downstream RAG components to gauge cognitive state uncertainty.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.mixture import GaussianMixture


def build_gmm_model(n_clusters: int = 5, random_state: int = 42) -> GaussianMixture:
    """Builds and returns an unconstrained full covariance GMM estimator."""
    return GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-5,
        random_state=random_state
    )


def get_gmm_centroids(model: GaussianMixture) -> np.ndarray:
    """Extracts mean centroid vectors mu_k of the fitted Gaussian components."""
    return model.means_
