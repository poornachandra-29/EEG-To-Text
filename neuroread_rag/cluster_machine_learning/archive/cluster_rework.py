"""
Clustering Architecture & BIC Model Selection Module.
Performs GMM clustering and model selection based on Bayesian Information Criterion (BIC):
- Evaluates candidate cluster configurations using standard BIC minimization.
- Preserves full cluster assignments for all sentences (no indeterminate gating).
- Integrates seamlessly with benchmark_results.csv.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "neuroread_rag" / "artifacts"
FIGURES_DIR = BASE_DIR / "figures"
STAGES_DIR = ARTIFACTS_DIR / "stages"
COHORT_CSV = STAGES_DIR / "stage3_cohort_processed.csv"

PRUNED_CLUSTER_FEATURES = [
    "EEI_rz",
    "theta_fm_rz",
    "BetaPower_rz",
    "AlphaPower_rz",
    "Theta_Alpha_Ratio_rz",
    "FFD_rz",
    "PupilSize_rz",
    "GD_rz",
    "TRT_rz",
    "FixationCount_rz",
]

CLUSTER_NAMES = {
    0: "Cluster 0 (Fluent Flow State)",
    1: "Cluster 1 (Deep Focus / Processing)",
    2: "Cluster 2 (Comprehension Breakdown)",
    3: "Cluster 3 (Cognitive Saturation)",
    4: "Cluster 4 (Attentional Drift)",
    5: "Cluster 5 (Normal Reading)",
}


class ClusterReworkPipeline:
    """
    Standard GMM clustering with direct BIC calculation and full sentence assignment.
    """

    def __init__(self, cohort_csv_path: Optional[Path] = None):
        self.cohort_path = Path(cohort_csv_path or COHORT_CSV)
        self.pca_: Optional[PCA] = None
        self.scaler_: Optional[StandardScaler] = None
        self.imputer_: Optional[SimpleImputer] = None
        self.selected_gmm_: Optional[GaussianMixture] = None

    def run_sweep_and_fit(self, k_range: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run standard BIC sweep across cluster counts, select model minimizing BIC, and save artifacts."""
        if not self.cohort_path.exists():
            raise FileNotFoundError(f"Cohort CSV not found at {self.cohort_path}")

        df = pd.read_csv(self.cohort_path)
        valid_feats = [f for f in PRUNED_CLUSTER_FEATURES if f in df.columns]

        X_raw = df[valid_feats].values.astype(float)
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_raw)

        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X_imp)

        # PCA retaining >= 90% variance
        self.pca_ = PCA(n_components=0.90, random_state=42)
        X_pca = self.pca_.fit_transform(X_std)
        explained_var = float(np.sum(self.pca_.explained_variance_ratio_))

        candidates = k_range or list(range(2, 9))
        sweep_records = []
        cov_types = ["full", "diag", "tied"]

        for cov in cov_types:
            for k in candidates:
                gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, n_init=5)
                labels = gmm.fit_predict(X_pca)
                bic = float(gmm.bic(X_pca))
                aic = float(gmm.aic(X_pca))
                sil = float(silhouette_score(X_pca, labels)) if len(np.unique(labels)) > 1 else 0.0

                sweep_records.append({
                    "k": k,
                    "covariance_type": cov,
                    "bic": round(bic, 2),
                    "aic": round(aic, 2),
                    "silhouette": round(sil, 4),
                })

        sweep_df = pd.DataFrame(sweep_records)

        # Save cluster selection sweep report with BIC
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sweep_csv_path = REPORTS_DIR / "cluster_selection.csv"
        sweep_df.to_csv(sweep_csv_path, index=False)

        # Select model minimizing BIC
        best_row = sweep_df.sort_values("bic", ascending=True).iloc[0]
        best_k = int(best_row["k"])
        best_cov = str(best_row["covariance_type"])

        # Fit final selected GMM
        self.selected_gmm_ = GaussianMixture(n_components=best_k, covariance_type=best_cov, random_state=42, n_init=10)
        final_labels = self.selected_gmm_.fit_predict(X_pca)
        posteriors = self.selected_gmm_.predict_proba(X_pca)

        # Save model artifacts
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "gmm": self.selected_gmm_,
            "pca": self.pca_,
            "scaler": self.scaler_,
            "imputer": self.imputer_,
            "features": valid_feats,
            "best_k": best_k,
            "best_cov": best_cov,
            "bic": float(best_row["bic"]),
            "aic": float(best_row["aic"]),
        }, ARTIFACTS_DIR / "gmm_model_rework.joblib")

        # Cluster profiles without indeterminate gating (all sentences assigned cleanly)
        profiles = {}
        for c_id in range(best_k):
            c_mask = final_labels == c_id
            c_df = df[c_mask]
            feat_stats = {}
            for f in valid_feats:
                feat_stats[f] = {
                    "mean": round(float(c_df[f].mean()), 4),
                    "median": round(float(c_df[f].median()), 4),
                    "std": round(float(c_df[f].std()), 4),
                }

            profiles[c_id] = {
                "name": CLUSTER_NAMES.get(c_id, f"Cluster {c_id}"),
                "sentence_count": int(np.sum(c_mask)),
                "percentage": round(float(np.mean(c_mask) * 100.0), 2),
                "feature_profiles": feat_stats,
            }

        meta = {
            "selected_k": best_k,
            "selected_covariance": best_cov,
            "explained_variance_ratio": round(explained_var, 4),
            "model_bic": float(best_row["bic"]),
            "model_aic": float(best_row["aic"]),
            "silhouette": float(best_row["silhouette"]),
            "cluster_profiles": profiles,
        }

        profiles_json_path = ARTIFACTS_DIR / "cluster_profiles.json"
        with open(profiles_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta


def run_cluster_rework():
    pipeline = ClusterReworkPipeline()
    return pipeline.run_sweep_and_fit()


if __name__ == "__main__":
    run_cluster_rework()
