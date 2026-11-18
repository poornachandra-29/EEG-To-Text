"""
train_cluster_models.py — NeuroRead-RAG Clustering Model Training & Internal Benchmarking
========================================================================================
Trains multiple unsupervised clustering algorithms on the 11-participant training cohort
using 10 physiological brain/eye-tracking features.

Strictly adheres to unsupervised learning principles:
- Evaluates models using intrinsic geometric cluster validation metrics:
    * Silhouette Score (cohesion vs separation, [-1, 1], higher is better)
    * Calinski-Harabasz Index (variance ratio criterion, higher is better)
    * Davies-Bouldin Index (cluster similarity ratio, lower is better)
    * Bayesian Information Criterion (BIC, lower is better) [for GMMs]
    * Akaike Information Criterion (AIC, lower is better) [for GMMs]
- Assigns neurophysiologically grounded cognitive state names to each of the 5 clusters
  based on centroid feature profiles (Concentrated, Deep Focus, Fatigue, Workload, Bottleneck).
- Saves trained models, scalers, imputers, and benchmark metrics for downstream evaluation.
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings("ignore")

# Default 10 physiological features
DEFAULT_10_FEATURES = [
    "EEI",            # Engagement / Executive Index: Beta / (Alpha + Theta)
    "FMT",            # Frontal Midline Theta (4-8 Hz)
    "ASM",            # Alpha Spectral Asymmetry (8-13 Hz)
    "GD",             # Gaze Duration (ms)
    "TRT",            # Total Reading Time (ms)
    "FFD",            # First Fixation Duration (ms)
    "PupilSize",      # Mean Pupil Diameter
    "FixationCount",  # Total fixations on sentence
    "AlphaPower",     # Parietal-occipital Alpha band power (8-13 Hz)
    "BetaPower"       # Frontal-central Beta band power (13-30 Hz)
]

DEFAULT_TRAIN_PIDS = [
    "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN",
    "ZJS", "ZKB", "ZKH", "ZKW", "ZMG"
]
DEFAULT_HELDOUT_PID = "ZPH"


def parse_args():
    parser = argparse.ArgumentParser(description="Train unsupervised clustering models for NeuroRead-RAG.")
    parser.add_argument("--data_dir", type=str, default="neuroread_rag/data/precomputed_csvs/task2_NR",
                        help="Directory containing participant feature CSV files.")
    parser.add_argument("--output_dir", type=str, default="artifacts/cluster_models",
                        help="Directory to save trained model artifacts.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters (default: 5).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--heldout_pid", type=str, default=DEFAULT_HELDOUT_PID,
                        help="Held-out participant ID to exclude from training.")
    return parser.parse_args()


def load_training_data(data_dir: str, heldout_pid: str, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Loads all participant CSV files in data_dir excluding the held-out participant."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

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

    if not train_dfs:
        raise ValueError(f"No training data found after excluding held-out participant {heldout_pid}")

    combined_df = pd.concat(train_dfs, ignore_index=True)
    return combined_df, included_pids


def map_cluster_to_cognitive_state(centroid: np.ndarray, feature_names: List[str]) -> Tuple[str, str]:
    """
    Interprets and names a cluster based on its standardized feature centroid vector.
    Features: [EEI, FMT, ASM, GD, TRT, FFD, PupilSize, FixationCount, AlphaPower, BetaPower]
    
    Returns:
        (cluster_name, description)
    """
    f_dict = dict(zip(feature_names, centroid))
    eei = f_dict.get("EEI", 0.0)
    fmt = f_dict.get("FMT", 0.0)
    trt = f_dict.get("TRT", 0.0)
    gd = f_dict.get("GD", 0.0)
    pupil = f_dict.get("PupilSize", 0.0)
    fix_count = f_dict.get("FixationCount", 0.0)
    alpha = f_dict.get("AlphaPower", 0.0)
    beta = f_dict.get("BetaPower", 0.0)

    # Heuristic scoring based on established reading cognitive neuroscience
    # 1. Cognitive Fatigue: Depressed EEI + Elevated Alpha/Theta + Pupil Constriction
    fatigue_score = (-eei * 1.5) + (alpha * 1.0) + (fmt * 0.8) - (pupil * 0.8)
    
    # 2. Concentrated Fluent Reading: High EEI + Swift Eye Movements (low GD/TRT)
    concentrated_fluent_score = (eei * 1.5) - (trt * 0.8) - (gd * 0.8) - (alpha * 0.5)
    
    # 3. Deep Cognitive Processing: High EEI + Deliberate Reading (longer TRT/GD)
    deep_processing_score = (eei * 1.2) + (trt * 0.8) + (gd * 0.8) + (fmt * 0.5)
    
    # 4. High Workload / Cognitive Strain: High Theta + High Beta + Dilated Pupil
    workload_score = (fmt * 1.5) + (pupil * 1.2) + (beta * 1.0) + (alpha * 0.5)
    
    # 5. Comprehension Bottleneck: Extremely long TRT + High Fixation Count (Regressions)
    bottleneck_score = (trt * 1.5) + (fix_count * 1.3) + (gd * 1.0) - (eei * 0.5)

    scores = {
        "Concentrated Fluent Reading": concentrated_fluent_score,
        "Deep Cognitive Processing": deep_processing_score,
        "Cognitive Fatigue": fatigue_score,
        "High Cognitive Workload": workload_score,
        "Comprehension Bottleneck": bottleneck_score
    }
    
    best_name = max(scores, key=scores.get)
    
    descriptions = {
        "Concentrated Fluent Reading": "High executive engagement (positive EEI) with rapid, fluent eye movements; efficient semantic parsing.",
        "Deep Cognitive Processing": "Sustained high engagement (positive EEI) with deliberate, analytical reading pace and focused fixation duration.",
        "Cognitive Fatigue": "Depressed executive index (negative EEI), elevated alpha/theta band power, and autonomic pupil constriction indicating mental exhaustion.",
        "High Cognitive Workload": "Elevated frontal midline theta and dilated pupils indicating heavy working memory demand and mental strain.",
        "Comprehension Bottleneck": "Prolonged total reading times with high fixation counts and frequent saccadic regressions; lexical/syntactic struggle."
    }
    
    return best_name, descriptions[best_name]


def assign_unique_cluster_names(centroids: np.ndarray, feature_names: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Assigns unique, distinctive cognitive state names to each of the K clusters.
    Ensures that names are ranked and assigned uniquely across all 5 clusters.
    """
    k = centroids.shape[0]
    candidate_states = [
        ("Concentrated Fluent Reading", "High executive engagement (positive EEI) with rapid, fluent eye movements; efficient semantic parsing."),
        ("Deep Cognitive Processing", "Sustained high engagement (positive EEI) with deliberate, analytical reading pace and focused fixation duration."),
        ("Cognitive Fatigue", "Depressed executive index (negative EEI), elevated alpha/theta band power, and autonomic pupil constriction indicating mental exhaustion."),
        ("High Cognitive Workload", "Elevated frontal midline theta and dilated pupils indicating heavy working memory demand and mental strain."),
        ("Comprehension Bottleneck", "Prolonged total reading times with high fixation counts and frequent saccadic regressions; lexical/syntactic struggle.")
    ]

    # Calculate affinity scores between each cluster and each cognitive state
    affinity_matrix = np.zeros((k, len(candidate_states)))
    for i in range(k):
        f = dict(zip(feature_names, centroids[i]))
        eei = f.get("EEI", 0.0)
        fmt = f.get("FMT", 0.0)
        trt = f.get("TRT", 0.0)
        gd = f.get("GD", 0.0)
        pupil = f.get("PupilSize", 0.0)
        fix_count = f.get("FixationCount", 0.0)
        alpha = f.get("AlphaPower", 0.0)
        beta = f.get("BetaPower", 0.0)

        # 0: Concentrated Fluent
        affinity_matrix[i, 0] = (eei * 1.5) - (trt * 0.8) - (gd * 0.8) - (alpha * 0.5)
        # 1: Deep Cognitive Processing
        affinity_matrix[i, 1] = (eei * 1.2) + (trt * 0.8) + (gd * 0.8) + (fmt * 0.5)
        # 2: Cognitive Fatigue
        affinity_matrix[i, 2] = (-eei * 1.5) + (alpha * 1.0) + (fmt * 0.8) - (pupil * 0.8)
        # 3: High Cognitive Workload
        affinity_matrix[i, 3] = (fmt * 1.5) + (pupil * 1.2) + (beta * 1.0) + (alpha * 0.5)
        # 4: Comprehension Bottleneck
        affinity_matrix[i, 4] = (trt * 1.5) + (fix_count * 1.3) + (gd * 1.0) - (eei * 0.5)

    # Greedily assign best unique match
    assigned_states = {}
    used_states = set()
    
    # Sort pairs of (cluster, state) by score descending
    pairs = []
    for c in range(k):
        for s in range(len(candidate_states)):
            pairs.append((c, s, affinity_matrix[c, s]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    cluster_assigned = set()
    for c, s, score in pairs:
        if c not in cluster_assigned and s not in used_states:
            cluster_assigned.add(c)
            used_states.add(s)
            state_name, desc = candidate_states[s]
            is_concentrated = (s in [0, 1])
            is_fatigued = (s == 2)
            assigned_states[c] = {
                "name": state_name,
                "description": desc,
                "category": "Concentrated" if is_concentrated else ("Fatigued" if is_fatigued else "Workload/Bottleneck"),
                "affinity_score": round(float(score), 3),
                "centroid": {fn: round(float(centroids[c, idx]), 3) for idx, fn in enumerate(feature_names)}
            }
        if len(cluster_assigned) == k:
            break

    # Fallback if any cluster unassigned
    for c in range(k):
        if c not in assigned_states:
            state_name, desc = candidate_states[c % len(candidate_states)]
            assigned_states[c] = {
                "name": f"Cognitive State {c} ({state_name})",
                "description": desc,
                "category": "Workload/Bottleneck",
                "affinity_score": 0.0,
                "centroid": {fn: round(float(centroids[c, idx]), 3) for idx, fn in enumerate(feature_names)}
            }

    return assigned_states


def train_and_benchmark(data_dir: str, output_dir: str, n_clusters: int = 5, random_state: int = 42, heldout_pid: str = DEFAULT_HELDOUT_PID):
    print("=" * 80)
    print("NEUROREAD-RAG: CLUSTERING MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)
    print(f"Data Directory:    {data_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Cluster Cardinality (K): {n_clusters}")
    print(f"Held-Out Patient:  {heldout_pid}")
    print(f"Features (10):     {', '.join(DEFAULT_10_FEATURES)}")
    print("-" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    alt_out_path = Path("neuroread_rag/cluster_machine_learning/trained_models")
    alt_out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df, train_pids = load_training_data(data_dir, heldout_pid, DEFAULT_10_FEATURES)
    print(f"Loaded {len(train_df)} training sentences across {len(train_pids)} participants: {', '.join(train_pids)}")

    # 2. Preprocessing
    # Impute missing values with training set medians
    imputer = SimpleImputer(strategy="median")
    X_train_raw = train_df[DEFAULT_10_FEATURES].values
    X_train_imp = imputer.fit_transform(X_train_raw)

    # Robust scaling (median and interquartile range)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    # Save feature medians and scalers
    feature_medians = dict(zip(DEFAULT_10_FEATURES, imputer.statistics_.tolist()))
    joblib.dump(imputer, out_path / "train_imputer.joblib")
    joblib.dump(scaler, out_path / "train_robust_scaler.joblib")
    joblib.dump(feature_medians, out_path / "feature_medians.joblib")
    
    # Save copies to alternate location for convenience
    joblib.dump(imputer, alt_out_path / "train_imputer.joblib")
    joblib.dump(scaler, alt_out_path / "train_robust_scaler.joblib")
    joblib.dump(feature_medians, alt_out_path / "feature_medians.joblib")

    # 3. Define Candidate Models
    models: Dict[str, Any] = {
        "GMM (Full Covariance)": GaussianMixture(
            n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5, random_state=random_state
        ),
        "GMM (Tied Covariance)": GaussianMixture(
            n_components=n_clusters, covariance_type="tied", n_init=10, reg_covar=1e-5, random_state=random_state
        ),
        "GMM (Diag Covariance)": GaussianMixture(
            n_components=n_clusters, covariance_type="diag", n_init=10, reg_covar=1e-5, random_state=random_state
        ),
        "K-Means (k-means++)": KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state
        ),
        "Agglomerative (Ward)": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ),
        "BIRCH": Birch(
            n_clusters=n_clusters, threshold=0.5, branching_factor=50
        )
    }

    results = []
    cluster_profiles_all = {}
    saved_model_paths = {}

    print("\nFitting and evaluating clustering models on training cohort...")
    print(f"{'Algorithm':<26} | {'Silhouette':<10} | {'Calinski-H':<10} | {'Davies-B':<10} | {'BIC':<10} | {'AIC':<10} | {'Fit Time (s)':<12}")
    print("-" * 96)

    for name, model in models.items():
        t0 = time.time()
        labels = model.fit_predict(X_train_scaled)
        fit_time = time.time() - t0

        # Compute unsupervised cluster quality metrics
        sil = float(silhouette_score(X_train_scaled, labels))
        ch = float(calinski_harabasz_score(X_train_scaled, labels))
        db = float(davies_bouldin_score(X_train_scaled, labels))

        # Model-specific metrics
        bic = float(model.bic(X_train_scaled)) if hasattr(model, "bic") else np.nan
        aic = float(model.aic(X_train_scaled)) if hasattr(model, "aic") else np.nan
        inertia = float(model.inertia_) if hasattr(model, "inertia_") else np.nan

        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("-", "_")
        model_fname = f"{safe_name}.joblib"
        joblib.dump(model, out_path / model_fname)
        joblib.dump(model, alt_out_path / model_fname)
        saved_model_paths[name] = model_fname

        # Compute cluster centroids
        centroids = np.zeros((n_clusters, len(DEFAULT_10_FEATURES)))
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centroids[k] = X_train_scaled[mask].mean(axis=0)
            else:
                centroids[k] = 0.0

        # If GMM, we can also use model.means_
        if hasattr(model, "means_"):
            centroids = model.means_
        elif hasattr(model, "cluster_centers_"):
            centroids = model.cluster_centers_

        cluster_naming = assign_unique_cluster_names(centroids, DEFAULT_10_FEATURES)
        
        # Calculate cluster percentages on train set
        counts = pd.Series(labels).value_counts(normalize=True).sort_index() * 100
        for k in range(n_clusters):
            cluster_naming[k]["train_percentage"] = round(float(counts.get(k, 0.0)), 2)
            cluster_naming[k]["train_count"] = int((labels == k).sum())

        cluster_profiles_all[name] = cluster_naming

        results.append({
            "algorithm": name,
            "n_clusters": n_clusters,
            "silhouette_score": round(sil, 4),
            "calinski_harabasz_index": round(ch, 2),
            "davies_bouldin_index": round(db, 4),
            "bic": round(bic, 1) if not np.isnan(bic) else None,
            "aic": round(aic, 1) if not np.isnan(aic) else None,
            "inertia": round(inertia, 1) if not np.isnan(inertia) else None,
            "fit_time_sec": round(fit_time, 4),
            "model_file": model_fname
        })

        bic_str = f"{bic:10.1f}" if not np.isnan(bic) else "       N/A"
        aic_str = f"{aic:10.1f}" if not np.isnan(aic) else "       N/A"
        print(f"{name:<26} | {sil:<10.4f} | {ch:<10.2f} | {db:<10.4f} | {bic_str} | {aic_str} | {fit_time:<12.4f}")

    # 4. Save Benchmark Results
    results_df = pd.DataFrame(results)
    results_csv = out_path / "train_benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)
    results_df.to_csv(alt_out_path / "train_benchmark_results.csv", index=False)
    results_df.to_csv("neuroread_rag/cluster_machine_learning/benchmark_results.csv", index=False)

    # Save Cluster Profiles
    with open(out_path / "cluster_profiles.json", "w") as f:
        json.dump(cluster_profiles_all, f, indent=2)
    with open(alt_out_path / "cluster_profiles.json", "w") as f:
        json.dump(cluster_profiles_all, f, indent=2)

    # Save Training Manifest
    manifest = {
        "train_participants": train_pids,
        "n_train_sentences": len(train_df),
        "heldout_participant": heldout_pid,
        "features": DEFAULT_10_FEATURES,
        "n_clusters": n_clusters,
        "models": saved_model_paths,
        "feature_medians": feature_medians
    }
    with open(out_path / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(alt_out_path / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 5. Display Cluster Profiles for the Baseline System (GMM Full Covariance)
    print("\n" + "=" * 80)
    print("COGNITIVE STATE CLUSTER PROFILES (GMM Full Covariance Baseline)")
    print("=" * 80)
    baseline_naming = cluster_profiles_all["GMM (Full Covariance)"]
    for k in range(n_clusters):
        c_info = baseline_naming[k]
        print(f"Cluster {k}: {c_info['name'].upper()} [{c_info['category']}]")
        print(f"  Distribution: {c_info['train_percentage']}% of training sentences ({c_info['train_count']} sentences)")
        print(f"  Description:  {c_info['description']}")
        print(f"  Key Centroid Values (Z-scores):")
        cent = c_info['centroid']
        print(f"    EEI: {cent['EEI']:+.2f} | FMT: {cent['FMT']:+.2f} | Alpha: {cent['AlphaPower']:+.2f} | GD: {cent['GD']:+.2f} | TRT: {cent['TRT']:+.2f} | Pupil: {cent['PupilSize']:+.2f}")
        print()

    print(f"Training completed successfully! Artifacts written to: {out_path.resolve()}")
    return results_df, cluster_profiles_all


if __name__ == "__main__":
    args = parse_args()
    train_and_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        heldout_pid=args.heldout_pid
    )
