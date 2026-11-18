"""
train.py — Modular Multi-Algorithm Training & Cohort Benchmarking
================================================================
Trains 6 distinct clustering algorithms on the 11-participant training cohort
using 10 multimodal brain/eye-tracking features:
    1. GMM                      -> models/gmm_model.py
    2. HDBSCAN                  -> models/hdbscan_model.py
    3. K-Means                  -> models/kmeans_model.py
    4. Fuzzy C-Means            -> models/fuzzy_c_means_model.py
    5. Agglomerative Clustering -> models/agglomerative_model.py
    6. BIRCH                    -> models/birch_model.py

Each model is encapsulated in its own dedicated Python (.py) file in `models/`,
making it clean and modular for explaining to external faculty.

Unsupervised evaluation criteria computed on training data:
    - Silhouette Score (S in [-1, 1], higher is better)
    - Calinski-Harabasz Index (CH in [0, inf), higher is better)
    - Davies-Bouldin Index (DB in [0, inf), lower is better)
    - Fit Time (seconds)
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add current directory and submodules to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "models"))
sys.path.insert(0, str(CURRENT_DIR / "cluster_mapping"))

# Import individual model wrappers from models/
from gmm_model import build_gmm_model, get_gmm_centroids
from hdbscan_model import build_hdbscan_model, get_hdbscan_centroids
from kmeans_model import build_kmeans_model, get_kmeans_centroids
from fuzzy_c_means_model import build_fuzzy_c_means_model, get_fuzzy_c_means_centroids, FuzzyCMeansModel
from agglomerative_model import build_agglomerative_model, get_agglomerative_centroids
from birch_model import build_birch_model, get_birch_centroids

# Import cluster mapping logic
from cluster_mapper import assign_cognitive_states, compute_sentence_percentages

warnings.filterwarnings("ignore")

FEATURES_10 = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount", "AlphaPower", "BetaPower"
]

DEFAULT_TRAIN_PIDS = [
    "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN",
    "ZJS", "ZKB", "ZKH", "ZKW", "ZMG"
]
DEFAULT_HELDOUT_PID = "ZPH"


def parse_args():
    parser = argparse.ArgumentParser(description="Train 6 clustering models on cohort EEG/eye-tracking data.")
    parser.add_argument("--data_dir", type=str, default="neuroread_rag/data/precomputed_csvs/task2_NR",
                        help="Path to folder containing participant CSV files.")
    parser.add_argument("--output_dir", type=str, default="artifacts/cluster_models",
                        help="Path to save trained models and preprocessors.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters (default: 5).")
    parser.add_argument("--heldout_pid", type=str, default=DEFAULT_HELDOUT_PID,
                        help="Participant ID to exclude from training (default: ZPH).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def load_training_data(data_dir: str, heldout_pid: str) -> Tuple[pd.DataFrame, List[str]]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(p.glob("features_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No features_*.csv found in {data_dir}")

    train_dfs = []
    included_pids = []

    for f in csv_files:
        df = pd.read_csv(f)
        pid = df["patient_name"].iloc[0] if "patient_name" in df.columns else f.stem.split("_")[-1]
        if pid == heldout_pid:
            continue
        train_dfs.append(df)
        included_pids.append(pid)

    combined_df = pd.concat(train_dfs, ignore_index=True)
    return combined_df, included_pids


def train_pipeline(data_dir: str, output_dir: str, n_clusters: int = 5, heldout_pid: str = DEFAULT_HELDOUT_PID, random_state: int = 42):
    print("=" * 80)
    print("NEUROREAD-RAG: MODULAR CLUSTERING MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Data Directory:         {data_dir}")
    print(f"Output Artifacts Dir:   {output_dir}")
    print(f"Cluster Cardinality K:  {n_clusters}")
    print(f"Held-Out Participant:   {heldout_pid}")
    print(f"Features (10):          {', '.join(FEATURES_10)}")
    print("-" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    alt_models_path = CURRENT_DIR / "trained_models"
    alt_models_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Training Data
    train_df, train_pids = load_training_data(data_dir, heldout_pid)
    n_train = len(train_df)
    print(f"Loaded {n_train} training sentences across {len(train_pids)} participants: {', '.join(train_pids)}")

    # 2. Preprocessing: Imputation & Robust Scaling (Fitted strictly on Train cohort)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_raw = train_df[FEATURES_10].values
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    # Save preprocessing pipelines
    joblib.dump(imputer, out_path / "train_imputer.joblib")
    joblib.dump(scaler, out_path / "train_robust_scaler.joblib")
    joblib.dump(imputer, alt_models_path / "train_imputer.joblib")
    joblib.dump(scaler, alt_models_path / "train_robust_scaler.joblib")

    # 3. Instantiate Models by Calling Dedicated Modules in models/
    models_dict = {
        "GMM": (build_gmm_model(n_clusters=n_clusters, random_state=random_state), "gmm_model.py"),
        "HDBSCAN": (build_hdbscan_model(min_cluster_size=50, min_samples=10), "hdbscan_model.py"),
        "K-Means": (build_kmeans_model(n_clusters=n_clusters, random_state=random_state), "kmeans_model.py"),
        "Fuzzy C-Means": (build_fuzzy_c_means_model(n_clusters=n_clusters, m=1.3, random_state=random_state), "fuzzy_c_means_model.py"),
        "Agglomerative Clustering": (build_agglomerative_model(n_clusters=n_clusters), "agglomerative_model.py"),
        "BIRCH": (build_birch_model(n_clusters=n_clusters, threshold=0.5, branching_factor=50), "birch_model.py")
    }

    train_benchmark = []
    cluster_profiles_all = {}
    saved_model_files = {}

    print("\nFitting all 6 clustering algorithms on the training cohort...")
    print(f"{'Algorithm':<28} | {'Silhouette':<10} | {'Calinski-H':<10} | {'Davies-B':<10} | {'Fit Time (s)':<12} | {'Source Module':<20}")
    print("-" * 102)

    for name, (model, module_fname) in models_dict.items():
        t0 = time.time()
        labels = model.fit_predict(X_train_scaled)
        fit_time = time.time() - t0

        # Extract centroids using dedicated helper or model attribute
        if name == "GMM":
            centroids = get_gmm_centroids(model)
        elif name == "HDBSCAN":
            centroids = get_hdbscan_centroids(model, X_train_scaled, labels)
        elif name == "K-Means":
            centroids = get_kmeans_centroids(model)
        elif name == "Fuzzy C-Means":
            centroids = get_fuzzy_c_means_centroids(model)
        elif name == "Agglomerative Clustering":
            centroids = get_agglomerative_centroids(labels, X_train_scaled, n_clusters)
        elif name == "BIRCH":
            # For BIRCH, compute empirical centroids for the K clusters
            centroids = np.zeros((n_clusters, len(FEATURES_10)))
            for k in range(n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    centroids[k] = X_train_scaled[mask].mean(axis=0)

        # Compute intrinsic validation metrics (ignoring noise -1 if evaluating HDBSCAN)
        valid_mask = (labels >= 0) if -1 in labels else np.ones(len(labels), dtype=bool)
        if len(np.unique(labels[valid_mask])) > 1:
            sil = float(silhouette_score(X_train_scaled[valid_mask], labels[valid_mask]))
            ch = float(calinski_harabasz_score(X_train_scaled[valid_mask], labels[valid_mask]))
            db = float(davies_bouldin_score(X_train_scaled[valid_mask], labels[valid_mask]))
        else:
            sil, ch, db = np.nan, np.nan, np.nan

        # Map cluster centroids to neurophysiological states via Hungarian engine
        cluster_mapping = assign_cognitive_states(centroids, FEATURES_10)

        # Calculate cluster percentages
        pct_summary = compute_sentence_percentages(labels, cluster_mapping)
        for k in range(centroids.shape[0]):
            c_name = cluster_mapping[k]["name"]
            cluster_mapping[k]["train_percentage"] = pct_summary.get(c_name, 0.0)

        cluster_profiles_all[name] = cluster_mapping

        # Save model artifact
        safe_fname = f"{name.lower().replace(' ', '_').replace('-', '_')}.joblib"
        joblib.dump(model, out_path / safe_fname)
        joblib.dump(model, alt_models_path / safe_fname)
        saved_model_files[name] = safe_fname

        train_benchmark.append({
            "algorithm": name,
            "source_file": module_fname,
            "n_training_samples": n_train,
            "n_clusters": centroids.shape[0],
            "train_silhouette": round(sil, 4),
            "train_calinski_harabasz": round(ch, 2),
            "train_davies_bouldin": round(db, 4),
            "fit_time_sec": round(fit_time, 4),
            "model_file": safe_fname
        })

        print(f"{name:<28} | {sil:<10.4f} | {ch:<10.2f} | {db:<10.4f} | {fit_time:<12.4f} | {module_fname:<20}")

    # 4. Save Benchmark Tables & Manifest
    benchmark_df = pd.DataFrame(train_benchmark)
    benchmark_df.to_csv(out_path / "train_benchmark_results.csv", index=False)
    benchmark_df.to_csv(CURRENT_DIR / "benchmark_results.csv", index=False)

    with open(out_path / "cluster_profiles.json", "w") as f:
        json.dump(cluster_profiles_all, f, indent=2)
    with open(alt_models_path / "cluster_profiles.json", "w") as f:
        json.dump(cluster_profiles_all, f, indent=2)

    manifest = {
        "train_participants": train_pids,
        "n_training_samples": n_train,
        "heldout_participant": heldout_pid,
        "features": FEATURES_10,
        "n_clusters": n_clusters,
        "models": saved_model_files
    }
    with open(out_path / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(alt_models_path / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING FINISHED: All 6 algorithms fitted and saved successfully.")
    print(f"Artifacts directory: {out_path.resolve()}")
    print("=" * 80)
    return benchmark_df, cluster_profiles_all


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        heldout_pid=args.heldout_pid,
        random_state=args.random_state
    )
