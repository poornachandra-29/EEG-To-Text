"""
Optional MATLAB (.mat) File Processor for NeuroRead RAG.
Allows users to directly pass any participant .mat file (e.g. resultsZJS_TSR.mat).
Extracts sentence-level features, applies saved preprocessors & GMM models,
computes composite cognitive indicators, and outputs a standardized CSV.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
import json
import joblib
import numpy as np
import pandas as pd

from nrei_extractor import extract_patient_nrei, MODEL_FEATURES
from preprocess import transform_with_saved_preprocessor
from neuroread_rag.config import PROJECT_ROOT, OUTPUT_DIR, CSV_DIR

EPS = 1e-9


def extract_task_and_patient_info(mat_file: Path) -> Dict[str, str]:
    """
    Extracts normalized task name and patient name from .mat path.
    Example:
      '.../task3 - TSR/Matlab files/resultsZJS_TSR.mat'
      -> task_name: 'task3_TSR'
      -> patient_name: 'ZJS'
      -> patient_id: 'resultsZJS_TSR'
    """
    stem = mat_file.stem
    # 1. Task name resolution from parent directory or stem suffix
    parent_task_name = mat_file.parent.parent.name
    if "task" in parent_task_name.lower():
        task_clean = parent_task_name.replace(" ", "").replace("-", "_").strip("_")
        while "__" in task_clean:
            task_clean = task_clean.replace("__", "_")
        task_name = task_clean
    elif "_" in stem:
        suffix = stem.split("_")[-1]
        task_name = f"task_{suffix}"
    else:
        task_name = "task_reading"

    # 2. Patient name resolution
    if stem.lower().startswith("results"):
        after_prefix = stem[7:]  # strip 'results'
        patient_name = after_prefix.split("_")[0]
    elif "_" in stem:
        patient_name = stem.split("_")[0]
    else:
        patient_name = stem

    return {
        "task_name": task_name,
        "patient_name": patient_name,
        "patient_id": stem
    }


def robust_z(series: pd.Series) -> pd.Series:
    """Robust within-participant Z-score based on median and MAD."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(med):
        return pd.Series(np.nan, index=x.index, dtype=float)
    if pd.isna(mad) or mad < EPS:
        sd = x.std()
        if pd.isna(sd) or sd < EPS:
            return pd.Series(np.where(x.notna(), 0.0, np.nan), index=x.index, dtype=float)
        return (x - x.mean()) / sd
    return 0.67448975 * (x - med) / mad


def process_mat_file(
    mat_path: Path,
    artifact_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fs: float = 500.0
) -> Path:
    """
    Extracts features from a .mat file, applies trained GMM/scaler, computes
    cognitive indices, and returns the path to the ready-to-use CSV.
    """
    mat_file = Path(mat_path).expanduser().resolve()
    if not mat_file.is_file():
        raise FileNotFoundError(f"MATLAB file not found: {mat_file}")

    art_dir = Path(artifact_dir or (PROJECT_ROOT / "artifacts")).resolve()
    out_dir = Path(output_dir or CSV_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = extract_task_and_patient_info(mat_file)
    task_name = info["task_name"]
    patient_name = info["patient_name"]
    patient_id = info["patient_id"]

    print(f"[MatProcessor] Extracting features from {mat_file.name} (task: {task_name}, patient: {patient_name})...")

    # 1. Extract sentence-level NREI and eye-tracking features
    records = extract_patient_nrei(mat_file, fs=fs)
    if not records:
        raise RuntimeError(f"No sentence features extracted from {mat_file.name}")

    raw_df = pd.DataFrame(records)
    raw_df["patient_id"] = patient_id
    raw_df["patient_name"] = patient_name
    raw_df["task_name"] = task_name
    raw_df["source_file"] = mat_file.name

    # 2. Apply trained preprocessor
    scaler_file = art_dir / "robust_scaler.joblib"
    gmm_file = art_dir / "gmm_model.joblib"

    if scaler_file.exists() and gmm_file.exists():
        print(f"[MatProcessor] Applying saved preprocessor and GMM model from {art_dir}...")
        test_df, x_test = transform_with_saved_preprocessor(raw_df, art_dir)
        gmm_model = joblib.load(gmm_file)

        cluster_ids = gmm_model.predict(x_test)
        probabilities = gmm_model.predict_proba(x_test)
        max_prob = probabilities.max(axis=1)

        result_df = test_df.copy()
        result_df["cluster_id"] = cluster_ids
        result_df["cluster_name"] = [f"Cluster {i}" for i in cluster_ids]
        result_df["cluster_probability"] = max_prob
        result_df["prediction_status"] = np.where(
            max_prob >= 0.55,
            "Higher cluster membership confidence",
            "Uncertain - review"
        )
        for i in range(gmm_model.n_components):
            result_df[f"prob_cluster_{i}"] = probabilities[:, i]
    else:
        print("[MatProcessor] Warning: Trained GMM model not found. Proceeding with raw features.")
        result_df = raw_df.copy()
        result_df["cluster_id"] = 0
        result_df["cluster_name"] = "Cluster 0"
        result_df["cluster_probability"] = 1.0

    # 3. Add derived ratios and robust z-scores
    if "ThetaPower" in result_df.columns and "AlphaPower" in result_df.columns:
        result_df["Theta_Alpha_Ratio"] = result_df["ThetaPower"] / (result_df["AlphaPower"].abs() + EPS)
    if "ThetaPower" in result_df.columns and "BetaPower" in result_df.columns:
        result_df["Theta_Beta_Ratio"] = result_df["BetaPower"] / (result_df["ThetaPower"].abs() + EPS)

    core_features = ["EEI", "FMT", "ASM", "GD", "TRT", "FFD", "PupilSize", "FixationCount", "ThetaPower", "AlphaPower", "BetaPower"]
    for col in core_features:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
            result_df[f"{col}_rz"] = robust_z(result_df[col])

    # 4. Add rolling variability
    for source, target in [("EEI", "EEI_local_sd"), ("FFD", "FFD_local_sd"), ("TRT", "TRT_local_sd"), ("PupilSize", "Pupil_local_sd")]:
        if source in result_df.columns:
            result_df[target] = result_df[source].rolling(10, min_periods=3).std()
            result_df[f"{target}_rz"] = robust_z(result_df[target])

    # 5. Add composite cognitive indices (REI, CWI, SAI, MWLI, VSI)
    rei_cols = [c for c in ["EEI_rz", "FMT_rz", "PupilSize_rz"] if c in result_df.columns]
    result_df["REI"] = result_df[rei_cols].mean(axis=1, skipna=True) if rei_cols else 0.0

    cwi_cols = [c for c in ["ThetaPower_rz", "PupilSize_rz", "GD_rz", "TRT_rz"] if c in result_df.columns]
    result_df["CWI"] = result_df[cwi_cols].mean(axis=1, skipna=True) if cwi_cols else 0.0

    instability_cols = [c for c in ["EEI_local_sd_rz", "FFD_local_sd_rz", "TRT_local_sd_rz", "Pupil_local_sd_rz"] if c in result_df.columns]
    instability = result_df[instability_cols].mean(axis=1, skipna=True) if instability_cols else pd.Series(0.0, index=result_df.index)

    result_df["SAI"] = result_df["REI"] - instability

    reading_var_cols = [c for c in ["FFD_local_sd_rz", "TRT_local_sd_rz"] if c in result_df.columns]
    reading_var = result_df[reading_var_cols].mean(axis=1, skipna=True) if reading_var_cols else pd.Series(0.0, index=result_df.index)
    result_df["MWLI_proxy"] = -result_df.get("EEI_rz", 0.0) + reading_var
    result_df["VSI_proxy"] = -instability

    # 6. Add pattern evidence scores
    neg_eei = -result_df.get("EEI_rz", 0.0)
    neg_ffd_var = -result_df.get("FFD_local_sd_rz", 0.0)
    neg_trt_var = -result_df.get("TRT_local_sd_rz", 0.0)

    result_df["focus_like_evidence"] = (result_df.get("EEI_rz", 0.0) + neg_ffd_var + neg_trt_var) / 3.0
    result_df["fatigue_compatible_evidence"] = (result_df.get("ThetaPower_rz", 0.0) + result_df.get("AlphaPower_rz", 0.0) + neg_eei) / 3.0
    result_df["attention_instability_evidence"] = (neg_eei + result_df.get("EEI_local_sd_rz", 0.0) + result_df.get("FFD_local_sd_rz", 0.0) + result_df.get("TRT_local_sd_rz", 0.0)) / 4.0
    result_df["workload_evidence"] = result_df["CWI"]

    # Pattern assignment
    score_cols = ["focus_like_evidence", "fatigue_compatible_evidence", "attention_instability_evidence", "workload_evidence"]
    labels, strengths, top_s, sec_s = [], [], [], []

    for _, row in result_df[score_cols].iterrows():
        valid = {k: float(row[k]) for k in score_cols if pd.notna(row[k])}
        ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            labels.append("Insufficient data")
            strengths.append("Unavailable")
            top_s.append(np.nan)
            sec_s.append(np.nan)
            continue
        top_name, top_val = ranked[0]
        sec_val = ranked[1][1] if len(ranked) > 1 else -np.inf
        top_s.append(top_val)
        sec_s.append(sec_val if np.isfinite(sec_val) else np.nan)
        if top_val < 0.50:
            label = "Mixed / indeterminate"
        elif top_val - sec_val < 0.25:
            label = "Mixed / overlapping evidence"
        else:
            label = {
                "focus_like_evidence": "Engagement-supported / focused-like",
                "fatigue_compatible_evidence": "Fatigue-compatible EEG pattern",
                "attention_instability_evidence": "Attention-instability / disengagement-like",
                "workload_evidence": "High cognitive-workload-like pattern"
            }.get(top_name, "Mixed")
        labels.append(label)
        strengths.append(
            "Stronger heuristic evidence" if top_val >= 1.0
            else "Moderate heuristic evidence" if top_val >= 0.5
            else "Weak / no clear evidence"
        )

    result_df["cognitive_pattern"] = labels
    result_df["pattern_evidence_strength"] = strengths
    result_df["top_pattern_score"] = top_s
    result_df["second_pattern_score"] = sec_s

    # 7. Save output CSV in separate folder ending by task name, patient name
    result_df["task_name"] = task_name
    result_df["patient_name"] = patient_name

    output_csv_filename = f"features_{task_name}_{patient_name}.csv"
    output_csv_path = out_dir / output_csv_filename
    result_df.to_csv(output_csv_path, index=False)
    print(f"[MatProcessor] Successfully processed {len(result_df)} sentences -> Saved to: {output_csv_path}")

    # Backward compatibility: link or mirror for stem lookups (e.g. resultsZJS_TSR_features.csv)
    try:
        compat_local = out_dir / f"{patient_id}_features.csv"
        if not compat_local.exists() and compat_local != output_csv_path:
            compat_local.symlink_to(output_csv_filename)
    except Exception:
        pass

    try:
        compat_root = OUTPUT_DIR / f"{patient_id}_features.csv"
        if compat_root != output_csv_path:
            if compat_root.is_symlink() or compat_root.exists():
                compat_root.unlink()
            compat_root.symlink_to(output_csv_path)
    except Exception:
        pass

    return output_csv_path
