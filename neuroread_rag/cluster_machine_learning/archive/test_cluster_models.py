"""
test_cluster_models.py — NeuroRead-RAG Held-Out Test Evaluation & Cognitive Dashboard
====================================================================================
Evaluates trained unsupervised clustering models on a held-out test participant (e.g. ZPH, 250 sentences)
using the 10 physiological features.

Adheres strictly to unsupervised learning principles:
- No ground-truth labels or 'accuracy' metrics are reported.
- Evaluates out-of-sample cluster quality using intrinsic validation metrics:
    * Test Silhouette Score (cohesion vs separation, [-1, 1], higher is better)
    * Test Calinski-Harabasz Index (variance ratio criterion, higher is better)
    * Test Davies-Bouldin Index (cluster similarity ratio, lower is better)
    * Inference Latency (ms per sentence)
- Calculates the percentage of sentences classified into each of the 5 cognitive states:
    * Concentrated Fluent Reading (%)
    * Deep Cognitive Processing (%)
    * Cognitive Fatigue (%)
    * High Cognitive Workload (%)
    * Comprehension Bottleneck (%)
- Outputs summary metrics, CSV tables, and visualization charts for the student dashboard.
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

warnings.filterwarnings("ignore")

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

DEFAULT_TEST_PID = "ZPH"


def parse_args():
    parser = argparse.ArgumentParser(description="Test unsupervised clustering models on held-out participant.")
    parser.add_argument("--data_dir", type=str, default="neuroread_rag/data/precomputed_csvs/task2_NR",
                        help="Directory containing participant feature CSV files.")
    parser.add_argument("--models_dir", type=str, default="artifacts/cluster_models",
                        help="Directory containing trained model artifacts.")
    parser.add_argument("--test_pid", type=str, default=DEFAULT_TEST_PID,
                        help="Held-out participant ID to test (default: ZPH).")
    parser.add_argument("--output_dir", type=str, default="neuroread_rag/cluster_machine_learning",
                        help="Directory to save test evaluation outputs and figures.")
    return parser.parse_args()


def load_test_data(data_dir: str, test_pid: str) -> pd.DataFrame:
    p = Path(data_dir)
    test_files = list(p.glob(f"features_*{test_pid}*.csv"))
    if not test_files:
        raise FileNotFoundError(f"Could not find test CSV for participant {test_pid} in {data_dir}")
    df = pd.read_csv(test_files[0])
    return df


def generate_visualizations(test_results_df: pd.DataFrame, dist_df: pd.DataFrame, output_dir: Path, test_pid: str):
    """Saves formatted text/CSV summaries for dashboard visualization."""
    figures_dir = output_dir / "visualizations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = figures_dir / f"{test_pid}_cognitive_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Cognitive State Sentence Distribution for Student {test_pid}\n")
        f.write("=" * 60 + "\n")
        for _, row in dist_df.iterrows():
            f.write(f"Algorithm: {row['algorithm']}\n")
            f.write(f"  Concentrated Reading: {row['Total_Concentrated_Percentage']}%\n")
            f.write(f"  Cognitive Fatigue:    {row['Total_Fatigued_Percentage']}%\n")
            f.write(f"  High Workload:        {row['Total_Workload_Percentage']}%\n")
            f.write(f"  Bottleneck:           {row['Total_Bottleneck_Percentage']}%\n\n")
    return summary_path

    print(f"Generated visualization plots in: {figures_dir}")
    return chart1_path, chart2_path


def test_models_pipeline(data_dir: str, models_dir: str, test_pid: str = DEFAULT_TEST_PID, output_dir: str = "neuroread_rag/cluster_machine_learning"):
    print("=" * 80)
    print("NEUROREAD-RAG: HELD-OUT CLUSTERING EVALUATION & COGNITIVE DASHBOARD")
    print("=" * 80)
    print(f"Data Directory:       {data_dir}")
    print(f"Models Directory:     {models_dir}")
    print(f"Held-Out Student ID:  {test_pid}")
    print(f"Output Directory:     {output_dir}")
    print("-" * 80)

    models_path = Path(models_dir)
    if not models_path.exists():
        # Fallback to alternate models dir if not in artifacts
        alt_path = Path("neuroread_rag/cluster_machine_learning/trained_models")
        if alt_path.exists():
            models_path = alt_path
        else:
            raise FileNotFoundError(f"Trained models directory not found at {models_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessing Artifacts
    imputer_file = models_path / "train_imputer.joblib"
    scaler_file = models_path / "train_robust_scaler.joblib"
    profiles_file = models_path / "cluster_profiles.json"
    manifest_file = models_path / "train_manifest.json"

    if not imputer_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(f"Missing imputer or scaler in {models_path}. Please run train_cluster_models.py first.")

    imputer = joblib.load(imputer_file)
    scaler = joblib.load(scaler_file)
    
    with open(profiles_file, "r") as f:
        cluster_profiles = json.load(f)
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    # 2. Load Held-Out Test Data
    test_df = load_test_data(data_dir, test_pid)
    n_test = len(test_df)
    print(f"Loaded held-out test participant {test_pid}: {n_test} usable sentence trials.")

    # 3. Transform Test Data (Strict Isolation)
    X_test_raw = test_df[DEFAULT_10_FEATURES].values
    X_test_imp = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_imp)

    # 4. Evaluate Each Trained Model
    model_files = manifest["models"]
    test_metrics = []
    sentence_distributions = []
    detailed_sentence_predictions = {}

    print("\nEvaluating models on held-out test participant...")
    print(f"{'Algorithm':<26} | {'Test Sil':<10} | {'Test CH':<10} | {'Test DB':<10} | {'Concentrated %':<16} | {'Fatigued %':<12} | {'Infer (ms)':<10}")
    print("-" * 102)

    for algo_name, model_fname in model_files.items():
        m_path = models_path / model_fname
        if not m_path.exists():
            print(f"Warning: model artifact {m_path} not found, skipping...")
            continue
        model = joblib.load(m_path)
        
        # Predict on test data
        t0 = time.time()
        if hasattr(model, "predict"):
            preds = model.predict(X_test_scaled)
        else:
            # Nearest centroid assignment from cluster profiles
            profiles = cluster_profiles[algo_name]
            centroids = np.array([[profiles[str(k)]["centroid"][f] for f in DEFAULT_10_FEATURES] for k in range(5)])
            dists = np.linalg.norm(X_test_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            preds = np.argmin(dists, axis=1)
        infer_time_ms = (time.time() - t0) * 1000.0 / n_test

        # Compute unsupervised evaluation metrics
        n_unique = len(np.unique(preds))
        if n_unique > 1:
            sil = float(silhouette_score(X_test_scaled, preds))
            ch = float(calinski_harabasz_score(X_test_scaled, preds))
            db = float(davies_bouldin_score(X_test_scaled, preds))
        else:
            sil, ch, db = np.nan, np.nan, np.nan

        # Map cluster IDs to named cognitive states
        profiles = cluster_profiles[algo_name]
        counts = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
        
        dist_record = {"algorithm": algo_name, "participant_id": test_pid, "total_sentences": n_test}
        concentrated_pct = 0.0
        fatigued_pct = 0.0
        workload_pct = 0.0
        bottleneck_pct = 0.0

        for k in range(5):
            k_str = str(k)
            c_info = profiles.get(k_str, {})
            c_name = c_info.get("name", f"Cluster {k}")
            c_cat = c_info.get("category", "Other")
            pct = round(float(counts.get(k, 0.0)), 2)
            dist_record[c_name] = pct
            
            if "Concentrated" in c_name or "Deep Cognitive Processing" in c_name or c_cat == "Concentrated":
                concentrated_pct += pct
            elif "Fatigue" in c_name or c_cat == "Fatigued":
                fatigued_pct += pct
            elif "Workload" in c_name:
                workload_pct += pct
            elif "Bottleneck" in c_name:
                bottleneck_pct += pct

        dist_record["Total_Concentrated_Percentage"] = round(concentrated_pct, 2)
        dist_record["Total_Fatigued_Percentage"] = round(fatigued_pct, 2)
        dist_record["Total_Workload_Percentage"] = round(workload_pct, 2)
        dist_record["Total_Bottleneck_Percentage"] = round(bottleneck_pct, 2)
        sentence_distributions.append(dist_record)

        test_metrics.append({
            "algorithm": algo_name,
            "participant_id": test_pid,
            "test_sentences": n_test,
            "test_silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "test_calinski_harabasz": round(ch, 2) if not np.isnan(ch) else None,
            "test_davies_bouldin": round(db, 4) if not np.isnan(db) else None,
            "concentrated_percentage": round(concentrated_pct, 2),
            "fatigued_percentage": round(fatigued_pct, 2),
            "inference_ms_per_sentence": round(infer_time_ms, 3)
        })

        detailed_sentence_predictions[algo_name] = preds.tolist()

        sil_str = f"{sil:<10.4f}" if not np.isnan(sil) else "       N/A"
        ch_str = f"{ch:<10.2f}" if not np.isnan(ch) else "       N/A"
        db_str = f"{db:<10.4f}" if not np.isnan(db) else "       N/A"
        print(f"{algo_name:<26} | {sil_str} | {ch_str} | {db_str} | {concentrated_pct:>13.2f}% | {fatigued_pct:>9.2f}% | {infer_time_ms:<10.3f}")

    # 5. Save Outputs
    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df.to_csv(out_path / "test_benchmark_results.csv", index=False)
    test_metrics_df.to_csv(models_path / "test_benchmark_results.csv", index=False)

    dist_df = pd.DataFrame(sentence_distributions)
    dist_df.to_csv(out_path / "test_cluster_sentence_distribution.csv", index=False)
    dist_df.to_csv(models_path / "test_cluster_sentence_distribution.csv", index=False)

    # Save sentence-level predictions back to a enriched test CSV
    enriched_test_df = test_df.copy()
    for algo_name, p_list in detailed_sentence_predictions.items():
        col_name = "cluster_" + algo_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("-", "_")
        enriched_test_df[col_name] = p_list
    enriched_test_df.to_csv(out_path / f"test_{test_pid}_clustered_sentences.csv", index=False)

    # 6. Generate Figures
    generate_visualizations(test_metrics_df, dist_df, out_path, test_pid)

    # 7. Print Cognitive Dashboard Report for Held-Out Student
    print("\n" + "=" * 80)
    print(f"COGNITIVE READING DASHBOARD: STUDENT {test_pid} (HELD-OUT TRIAL)")
    print("=" * 80)
    baseline_dist = next(item for item in sentence_distributions if item["algorithm"] == "GMM (Full Covariance)")
    print(f"Evaluated Sentences: {n_test}")
    print(f"Model:               GMM (Full Covariance) [Baseline System]")
    print(f"Validation Quality:  Silhouette = {test_metrics_df.loc[0, 'test_silhouette']}, Davies-Bouldin = {test_metrics_df.loc[0, 'test_davies_bouldin']}")
    print("-" * 80)
    print(f"  * Concentrated Reading Sentences: {baseline_dist['Total_Concentrated_Percentage']:.1f}%")
    print(f"      - Concentrated Fluent:        {baseline_dist.get('Concentrated Fluent Reading', 0.0):.1f}%")
    print(f"      - Deep Cognitive Processing:  {baseline_dist.get('Deep Cognitive Processing', 0.0):.1f}%")
    print(f"  * Cognitive Fatigue Sentences:    {baseline_dist['Total_Fatigued_Percentage']:.1f}%")
    print(f"  * High Workload / Bottleneck:     {baseline_dist['Total_Workload_Percentage'] + baseline_dist['Total_Bottleneck_Percentage']:.1f}%")
    print("-" * 80)
    if baseline_dist['Total_Concentrated_Percentage'] >= 50.0:
        print(f"DIAGNOSTIC SUMMARY: Student {test_pid} demonstrated predominant cognitive engagement and focus ({baseline_dist['Total_Concentrated_Percentage']:.1f}% of sentences).")
    else:
        print(f"DIAGNOSTIC SUMMARY: Student {test_pid} showed elevated cognitive fatigue ({baseline_dist['Total_Fatigued_Percentage']:.1f}%) during natural reading.")
    print("=" * 80)

    return test_metrics_df, dist_df


if __name__ == "__main__":
    args = parse_args()
    test_models_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        test_pid=args.test_pid,
        output_dir=args.output_dir
    )
