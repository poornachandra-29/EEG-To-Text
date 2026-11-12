"""
Self-contained Feature Extraction and Preprocessing Module for NeuroRead RAG.
Operates independently within the neuroread_rag package without requiring external root modules.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
import joblib

EPSILON = 1e-8

MODEL_FEATURES = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount"
]

EXTRA_DIAGNOSTIC_COLUMNS = [
    "ThetaPower", "AlphaPower", "BetaPower",
    "ParticipantAlphaBaseline"
]


def _as_items(value):
    """Turn MATLAB scalar/array structures into a flat Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return list(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _field(obj, name, default=None):
    """Read a field from scipy mat_struct, numpy.void, or dict."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return default


def safe_float(value):
    """Convert scalar/array-like MATLAB values to a float; missing -> NaN."""
    if value is None:
        return np.nan
    try:
        arr = np.asarray(value, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else np.nan
    except (TypeError, ValueError):
        return np.nan


def compute_psd_bands(eeg_signal, fs=500.0) -> Tuple[float, float, float]:
    """Calculate mean channel-wise absolute band power using Welch PSD."""
    try:
        signal = np.asarray(eeg_signal, dtype=float)
        signal = np.squeeze(signal)

        if signal.ndim != 2:
            return np.nan, np.nan, np.nan

        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        if signal.shape[0] < 1 or signal.shape[1] < fs:
            return np.nan, np.nan, np.nan

        signal = np.where(np.isfinite(signal), signal, np.nan)
        valid_channels = np.all(np.isfinite(signal), axis=1)
        signal = signal[valid_channels]
        if signal.size == 0:
            return np.nan, np.nan, np.nan

        freqs, psd = welch(
            signal,
            fs=fs,
            axis=-1,
            nperseg=min(512, signal.shape[1])
        )

        def band_power(low, high):
            mask = (freqs >= low) & (freqs <= high)
            if np.count_nonzero(mask) < 2:
                return np.nan
            return float(np.mean(np.trapz(psd[:, mask], freqs[mask], axis=-1)))

        return band_power(4.0, 8.0), band_power(8.0, 13.0), band_power(13.0, 30.0)

    except Exception:
        return np.nan, np.nan, np.nan


def extract_patient_nrei(mat_path: Path, fs: float = 500.0) -> List[Dict[str, Any]]:
    """Extract sentence-level features from one participant MATLAB .mat file."""
    mat_path = Path(mat_path)
    try:
        mat = sio.loadmat(
            str(mat_path),
            struct_as_record=False,
            squeeze_me=True
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load {mat_path}: {exc}") from exc

    if "sentenceData" not in mat:
        raise KeyError(f"'sentenceData' was not found in {mat_path.name}")

    sentences = _as_items(mat["sentenceData"])
    parsed = []
    alpha_values = []

    for sentence_index, sentence in enumerate(sentences):
        content = _field(sentence, "content", "")
        raw_data = _field(sentence, "rawData", None)

        if raw_data is not None:
            theta, alpha, beta = compute_psd_bands(raw_data, fs=fs)
        else:
            theta, alpha, beta = np.nan, np.nan, np.nan

        if np.isfinite(alpha):
            alpha_values.append(alpha)

        parsed.append({
            "sentence_index": sentence_index,
            "sentence_text": str(content) if content is not None else "",
            "sentence_obj": sentence,
            "ThetaPower": theta,
            "AlphaPower": alpha,
            "BetaPower": beta,
        })

    baseline_alpha = float(np.median(alpha_values)) if alpha_values else np.nan
    records = []

    for item in parsed:
        theta = item["ThetaPower"]
        alpha = item["AlphaPower"]
        beta = item["BetaPower"]

        eei = (
            beta / (alpha + theta + EPSILON)
            if np.isfinite(beta) and np.isfinite(alpha) and np.isfinite(theta)
            else np.nan
        )
        fmt = theta
        asm = (
            (baseline_alpha - alpha) / (baseline_alpha + EPSILON)
            if np.isfinite(baseline_alpha) and np.isfinite(alpha)
            else np.nan
        )

        words = _as_items(_field(item["sentence_obj"], "word", []))
        gd, trt, ffd, pupil, fixations = [], [], [], [], []

        for word in words:
            for field_name, target in [
                ("GD", gd),
                ("TRT", trt),
                ("FFD", ffd),
                ("meanPupilSize", pupil),
                ("nFixations", fixations),
            ]:
                value = safe_float(_field(word, field_name, None))
                if np.isfinite(value):
                    target.append(value)

        def mean_or_nan(values):
            return float(np.mean(values)) if values else np.nan

        records.append({
            "sentence_index": item["sentence_index"],
            "sentence_text": item["sentence_text"],
            "EEI": eei,
            "FMT": fmt,
            "ASM": asm,
            "GD": mean_or_nan(gd),
            "TRT": mean_or_nan(trt),
            "FFD": mean_or_nan(ffd),
            "PupilSize": mean_or_nan(pupil),
            "FixationCount": mean_or_nan(fixations),
            "ThetaPower": theta,
            "AlphaPower": alpha,
            "BetaPower": beta,
            "ParticipantAlphaBaseline": baseline_alpha,
        })

    return records


def clean_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep metadata and model features; coerce invalid numeric values to NaN."""
    result = df.copy()
    for col in MODEL_FEATURES:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows only if every model feature is missing
    existing_cols = [c for c in MODEL_FEATURES if c in result.columns]
    if existing_cols:
        result = result.loc[result[existing_cols].notna().any(axis=1)].copy()
    return result


def transform_with_saved_preprocessor(df: pd.DataFrame, artifact_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply saved medians and RobustScaler without refitting."""
    artifact_dir = Path(artifact_dir)
    medians_file = artifact_dir / "feature_medians.joblib"
    scaler_file = artifact_dir / "robust_scaler.joblib"

    if not medians_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(f"Missing preprocessor artifacts in {artifact_dir}")

    medians = joblib.load(medians_file)
    scaler = joblib.load(scaler_file)

    df_cleaned = clean_feature_table(df)
    x_filled = df_cleaned[MODEL_FEATURES].fillna(pd.Series(medians))
    x_scaled = scaler.transform(x_filled)
    return df_cleaned, x_scaled
