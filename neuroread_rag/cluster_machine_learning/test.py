"""
test.py — Model Testing, Cognitive Dashboard & Best Model Selection
===================================================================
Evaluates the 6 trained clustering algorithms on held-out patient data:
    1. GMM                      -> models/gmm_model.py
    2. HDBSCAN                  -> models/hdbscan_model.py
    3. K-Means                  -> models/kmeans_model.py
    4. Fuzzy C-Means            -> models/fuzzy_c_means_model.py
    5. Agglomerative Clustering -> models/agglomerative_model.py
    6. BIRCH                    -> models/birch_model.py

Features evaluated (10):
    [EEI, FMT, ASM, GD, TRT, FFD, PupilSize, FixationCount, AlphaPower, BetaPower]

Metrics considered (Unsupervised Out-of-Sample Validation):
    - Silhouette Score (cohesion vs separation, [-1, 1], higher is better)
    - Calinski-Harabasz Index (variance ratio criterion, higher is better)
    - Davies-Bouldin Index (cluster similarity ratio, lower is better)
    - Inference Latency (ms per sentence trial)

Calculates the percentage of sentences in:
    - Concentrated Reading (Fluent + Deep Focus)
    - Cognitive Fatigue (Low Vigilance / Cortical Slowing)
    - Workload / Comprehension Bottlenecks

Selects the Best Model based on test set unsupervised validation metrics and saves results.
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "models"))
sys.path.insert(0, str(CURRENT_DIR / "cluster_mapping"))

from cluster_mapper import compute_sentence_percentages
from fuzzy_c_means_model import FuzzyCMeansModel
from hdbscan_model import predict_hdbscan
from agglomerative_model import predict_agglomerative

warnings.filterwarnings("ignore")

FEATURES_10 = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount", "AlphaPower", "BetaPower"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clustering models on a test patient.")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to specific test patient CSV file (e.g. features_task2_NR_ZPH.csv).")
    parser.add_argument("--test_dir", type=str, default="neuroread_rag/data/precomputed_csvs/task2_NR",
                        help="Path to directory containing test patient CSV files.")
    parser.add_argument("--test_pid", type=str, default="ZPH",
                        help="Participant ID to test if test_file is not specified (default: ZPH).")
    parser.add_argument("--models_dir", type=str, default="artifacts/cluster_models",
                        help="Path to saved models directory.")
    parser.add_argument("--output_dir", type=str, default="neuroread_rag/cluster_machine_learning",
                        help="Path to save test benchmark results.")
    return parser.parse_args()


def resolve_test_file(args) -> Tuple[Path, str]:
    if args.test_file:
        p = Path(args.test_file)
        if not p.exists():
            raise FileNotFoundError(f"Test file not found: {args.test_file}")
        pid = p.stem.split("_")[-1]
        return p, pid

    # Look in test_dir
    d = Path(args.test_dir)
    matches = list(d.glob(f"*{args.test_pid}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No test CSV found for participant {args.test_pid} in {args.test_dir}")
    return matches[0], args.test_pid


def select_best_model(metrics_df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Selects the best clustering model on the test set using a multi-criteria
    normalized composite score across Silhouette, Calinski-Harabasz, and Davies-Bouldin.
    """
    df = metrics_df.copy()

    sil = df["test_silhouette"].values
    ch = df["test_calinski_harabasz"].values
    db = df["test_davies_bouldin"].values

    def norm_high(v):
        v_min, v_max = np.nanmin(v), np.nanmax(v)
        return (v - v_min) / (v_max - v_min + 1e-9)

    def norm_low(v):
        v_min, v_max = np.nanmin(v), np.nanmax(v)
        return (v_max - v) / (v_max - v_min + 1e-9)

    norm_sil = norm_high(sil)
    norm_ch = norm_high(ch)
    norm_db = norm_low(db)

    # Weighted Composite Score: 40% Silhouette + 30% Calinski-Harabasz + 30% Davies-Bouldin
    composite_scores = 0.40 * norm_sil + 0.30 * norm_ch + 0.30 * norm_db
    df["composite_score"] = np.round(composite_scores, 4)

    best_idx = int(np.argmax(composite_scores))
    best_row = df.iloc[best_idx]
    best_model_name = best_row["algorithm"]

    selection_info = {
        "best_model": best_model_name,
        "composite_score": float(best_row["composite_score"]),
        "test_silhouette": float(best_row["test_silhouette"]),
        "test_calinski_harabasz": float(best_row["test_calinski_harabasz"]),
        "test_davies_bouldin": float(best_row["test_davies_bouldin"]),
        "concentrated_percentage": float(best_row["concentrated_percentage"]),
        "fatigued_percentage": float(best_row["fatigued_percentage"]),
        "all_rankings": df[["algorithm", "composite_score", "test_silhouette", "test_calinski_harabasz", "test_davies_bouldin"]].to_dict(orient="records")
    }

    return best_model_name, selection_info


def test_pipeline(args):
    test_csv_path, test_pid = resolve_test_file(args)
    models_path = Path(args.models_dir)
    if not models_path.exists():
        alt_path = CURRENT_DIR / "trained_models"
        if alt_path.exists():
            models_path = alt_path
        else:
            raise FileNotFoundError(f"Models directory not found at {models_path}. Run train.py first.")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NEUROREAD-RAG: CLUSTERING MODEL EVALUATION & SELECTION PIPELINE")
    print("=" * 80)
    print(f"Test File:           {test_csv_path}")
    print(f"Test Participant ID: {test_pid}")
    print(f"Models Directory:    {models_path}")
    print(f"Output Directory:    {out_path}")
    print("-" * 80)

    # 1. Load Preprocessing Pipelines & Manifest
    imputer = joblib.load(models_path / "train_imputer.joblib")
    scaler = joblib.load(models_path / "train_robust_scaler.joblib")

    with open(models_path / "train_manifest.json", "r") as f:
        manifest = json.load(f)
    with open(models_path / "cluster_profiles.json", "r") as f:
        cluster_profiles = json.load(f)

    # 2. Load and Preprocess Test Data
    test_df = pd.read_csv(test_csv_path)
    n_test = len(test_df)
    print(f"Loaded test dataset: {n_test} usable sentence trials for participant {test_pid}.")

    X_test_raw = test_df[FEATURES_10].values
    X_test_imp = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_imp)

    # 3. Evaluate Each Trained Algorithm
    model_files = manifest["models"]
    test_records = []
    sentence_dist_records = []
    sentence_predictions = {}

    print("\nEvaluating all 6 algorithms on held-out test patient...")
    print(f"{'Algorithm':<26} | {'Test Sil':<10} | {'Test CH':<10} | {'Test DB':<10} | {'Concentrated %':<16} | {'Fatigued %':<12} | {'Latency (ms)':<12}")
    print("-" * 104)

    for algo_name, model_file in model_files.items():
        m_path = models_path / model_file
        if not m_path.exists():
            print(f"Warning: {m_path} not found, skipping...")
            continue
        model = joblib.load(m_path)

        t0 = time.time()
        # Predict on test data
        if algo_name == "HDBSCAN":
            # Extract non-noise centroids
            centroids = model.centroids_
            preds = predict_hdbscan(model, centroids, X_test_scaled)
        elif algo_name == "Agglomerative Clustering":
            profiles = cluster_profiles[algo_name]
            centroids = np.array([[profiles[str(k)]["centroid"][f] for f in FEATURES_10] for k in range(manifest["n_clusters"])])
            preds = predict_agglomerative(centroids, X_test_scaled)
        elif hasattr(model, "predict"):
            preds = model.predict(X_test_scaled)
        else:
            profiles = cluster_profiles[algo_name]
            centroids = np.array([[profiles[str(k)]["centroid"][f] for f in FEATURES_10] for k in range(manifest["n_clusters"])])
            dists = np.linalg.norm(X_test_scaled[:, None, :] - centroids[None, :, :], axis=2)
            preds = np.argmin(dists, axis=1)

        infer_time_ms = ((time.time() - t0) * 1000.0) / n_test

        # Compute unsupervised evaluation metrics
        n_unique = len(np.unique(preds))
        if n_unique > 1:
            sil = float(silhouette_score(X_test_scaled, preds))
            ch = float(calinski_harabasz_score(X_test_scaled, preds))
            db = float(davies_bouldin_score(X_test_scaled, preds))
        else:
            sil, ch, db = np.nan, np.nan, np.nan

        # Map cluster IDs to cognitive states & compute sentence percentages
        profiles = cluster_profiles.get(algo_name, {})
        pct_dict = compute_sentence_percentages(preds, profiles)

        concentrated_pct = pct_dict.get("Total_Concentrated_Percentage", 0.0)
        fatigued_pct = pct_dict.get("Total_Fatigued_Percentage", 0.0)
        workload_pct = pct_dict.get("Total_Workload_Percentage", 0.0)
        bottleneck_pct = pct_dict.get("Total_Bottleneck_Percentage", 0.0)

        dist_entry = {
            "algorithm": algo_name,
            "participant_id": test_pid,
            "n_test_samples": n_test,
            "Total_Concentrated_Percentage": concentrated_pct,
            "Total_Fatigued_Percentage": fatigued_pct,
            "Total_Workload_Percentage": workload_pct,
            "Total_Bottleneck_Percentage": bottleneck_pct
        }
        for k in range(manifest["n_clusters"]):
            c_info = profiles.get(str(k), profiles.get(k, {}))
            c_name = c_info.get("name", f"Cluster {k}")
            dist_entry[c_name] = pct_dict.get(c_name, 0.0)
        sentence_dist_records.append(dist_entry)

        test_records.append({
            "algorithm": algo_name,
            "participant_id": test_pid,
            "n_test_samples": n_test,
            "test_silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "test_calinski_harabasz": round(ch, 2) if not np.isnan(ch) else None,
            "test_davies_bouldin": round(db, 4) if not np.isnan(db) else None,
            "concentrated_percentage": concentrated_pct,
            "fatigued_percentage": fatigued_pct,
            "latency_ms_per_sentence": round(infer_time_ms, 4)
        })

        sentence_predictions[algo_name] = preds.tolist()

        sil_str = f"{sil:<10.4f}" if not np.isnan(sil) else "       N/A"
        ch_str = f"{ch:<10.2f}" if not np.isnan(ch) else "       N/A"
        db_str = f"{db:<10.4f}" if not np.isnan(db) else "       N/A"
        print(f"{algo_name:<26} | {sil_str} | {ch_str} | {db_str} | {concentrated_pct:>13.2f}% | {fatigued_pct:>9.2f}% | {infer_time_ms:<12.4f}")

    # 4. Save CSV Outputs
    metrics_df = pd.DataFrame(test_records)
    metrics_df.to_csv(out_path / "test_benchmark_results.csv", index=False)
    metrics_df.to_csv(models_path / "test_benchmark_results.csv", index=False)

    dist_df = pd.DataFrame(sentence_dist_records)
    dist_df.to_csv(out_path / "test_cluster_sentence_distribution.csv", index=False)
    dist_df.to_csv(models_path / "test_cluster_sentence_distribution.csv", index=False)

    # Save sentence predictions
    pred_df = test_df.copy()
    for algo_name, p_list in sentence_predictions.items():
        col_name = "cluster_" + algo_name.lower().replace(" ", "_").replace("-", "_")
        pred_df[col_name] = p_list
    pred_df.to_csv(out_path / "test_predictions.csv", index=False)

    # 5. Model Selection
    best_model_name, selection_info = select_best_model(metrics_df)
    with open(out_path / "best_model_selection.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    # 6. Print Best Model Selection & Summary Report
    print("\n" + "=" * 80)
    print(f"BEST MODEL SELECTION RESULT (TEST PARTICIPANT: {test_pid})")
    print("=" * 80)
    print(f"CHAMPION MODEL:          {best_model_name.upper()}")
    print(f"Composite Score:         {selection_info['composite_score']:.4f} (Rank 1 / {len(model_files)})")
    print(f"Test Silhouette:         {selection_info['test_silhouette']:.4f}")
    print(f"Test Calinski-Harabasz:  {selection_info['test_calinski_harabasz']:.2f}")
    print(f"Test Davies-Bouldin:     {selection_info['test_davies_bouldin']:.4f}")
    print("-" * 80)
    print(f"Sentence Breakdown for Student {test_pid} by Champion Model ({best_model_name}):")
    print(f"  * Concentrated Reading Sentences: {selection_info['concentrated_percentage']:.1f}%")
    print(f"  * Cognitive Fatigue Sentences:    {selection_info['fatigued_percentage']:.1f}%")
    print("=" * 80)

    return metrics_df, dist_df, selection_info


if __name__ == "__main__":
    args = parse_args()
    test_pipeline(args)
