"""
Self-contained Feature Extraction and Preprocessing Module for NeuroRead RAG.
Implements Phase 1: Frontal Midline Theta (FMT), window reliability gating,
artifact detection, and log-transformed spectral powers.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
import joblib

# Configuration defaults (aligns with config.yaml)
EPSILON = 1e-8
FS_DEFAULT = 500.0
FRONTAL_MIDLINE_CHANNELS = [5, 10]  # E6 (FCz), E11 (Fz) in 105 scalp channel set
PEAK_TO_PEAK_THRESHOLD = 150.0  # µV amplitude threshold for artifact detection
ARTIFACT_THRESHOLD_FRACTION = 0.20

MODEL_FEATURES = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount"
]

EXTRA_DIAGNOSTIC_COLUMNS = [
    "ThetaPower", "AlphaPower", "BetaPower",
    "ParticipantAlphaBaseline", "theta_global", "theta_fm",
    "eeg_duration_s", "eeg_reliability", "artifact_fraction",
    "log_theta", "log_alpha", "log_beta"
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


def compute_eeg_reliability(signal_shape: Tuple[int, int], fs: float = 500.0) -> Tuple[str, int]:
    """
    Determine window reliability and Welch segment length based on duration.
    - Duration >= 4.0s (>= 2000 samples): 'full' reliability, nperseg=512.
    - Duration 2.0s to 4.0s (1000 to 2000 samples): 'low' reliability, nperseg=256.
    - Duration < 2.0s (< 1000 samples): 'insufficient' signal, nperseg=0 (skip).
    """
    n_samples = signal_shape[1] if signal_shape[0] < signal_shape[1] else signal_shape[0]
    duration_s = n_samples / fs
    if duration_s < 2.0:
        return "insufficient", 0
    elif duration_s < 4.0:
        return "low", 256
    else:
        return "full", 512


def detect_artifacts(signal: np.ndarray, threshold_uv: float = PEAK_TO_PEAK_THRESHOLD) -> float:
    """
    Calculate fraction of samples affected by amplitude artifacts (e.g. > threshold_uv).
    Returns fraction between 0.0 and 1.0.
    """
    if signal.size == 0:
        return 1.0
    ptp = np.ptp(signal, axis=-1)  # peak-to-peak per channel
    bad_channels = np.mean(ptp > threshold_uv)
    return float(bad_channels)


def compute_psd_bands(
    eeg_signal,
    fs: float = 500.0,
    frontal_channels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Calculate channel-wise and frontal-midline absolute band powers using Welch PSD.
    Implements sentence duration gating and artifact rejection.
    """
    out = {
        "theta_global": np.nan,
        "alpha_global": np.nan,
        "beta_global": np.nan,
        "theta_fm": np.nan,
        "eeg_duration_s": 0.0,
        "eeg_reliability": "insufficient",
        "artifact_fraction": 0.0,
        "log_theta": np.nan,
        "log_alpha": np.nan,
        "log_beta": np.nan
    }
    
    if eeg_signal is None:
        return out

    try:
        signal = np.asarray(eeg_signal, dtype=float)
        signal = np.squeeze(signal)

        if signal.ndim != 2:
            return out

        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        n_channels, n_samples = signal.shape
        duration_s = n_samples / fs
        out["eeg_duration_s"] = float(duration_s)

        reliability, nperseg = compute_eeg_reliability(signal.shape, fs=fs)
        out["eeg_reliability"] = reliability

        # Check amplitude artifact fraction
        artifact_frac = detect_artifacts(signal)
        out["artifact_fraction"] = artifact_frac
        if artifact_frac > ARTIFACT_THRESHOLD_FRACTION:
            out["eeg_reliability"] = "insufficient"

        if out["eeg_reliability"] == "insufficient" or nperseg == 0:
            return out

        signal = np.where(np.isfinite(signal), signal, np.nan)
        valid_channels = np.all(np.isfinite(signal), axis=1)
        if not np.any(valid_channels):
            return out

        freqs, psd = welch(
            signal,
            fs=fs,
            axis=-1,
            nperseg=min(nperseg, n_samples)
        )

        def band_power(low: float, high: float, channels_mask=None) -> float:
            f_mask = (freqs >= low) & (freqs <= high)
            if np.count_nonzero(f_mask) < 2:
                return np.nan
            subset_psd = psd if channels_mask is None else psd[channels_mask]
            if subset_psd.size == 0 or not np.any(np.isfinite(subset_psd)):
                return np.nan
            return float(np.nanmean(np.trapz(subset_psd[:, f_mask], freqs[f_mask], axis=-1)))

        th_glob = band_power(4.0, 8.0)
        al_glob = band_power(8.0, 13.0)
        be_glob = band_power(13.0, 30.0)

        out["theta_global"] = th_glob
        out["alpha_global"] = al_glob
        out["beta_global"] = be_glob

        if np.isfinite(th_glob) and th_glob > 0:
            out["log_theta"] = float(np.log10(th_glob))
        if np.isfinite(al_glob) and al_glob > 0:
            out["log_alpha"] = float(np.log10(al_glob))
        if np.isfinite(be_glob) and be_glob > 0:
            out["log_beta"] = float(np.log10(be_glob))

        # Frontal midline theta (FMT): channels nearest Fz/FCz (e.g. E6, E11)
        fm_channels = frontal_channels or FRONTAL_MIDLINE_CHANNELS
        valid_fm = [c for c in fm_channels if 0 <= c < n_channels and valid_channels[c]]
        if valid_fm:
            out["theta_fm"] = band_power(4.0, 8.0, channels_mask=valid_fm)
        else:
            out["theta_fm"] = th_glob

        return out

    except Exception:
        return out


def extract_patient_nrei(
    mat_path: Path,
    fs: float = 500.0,
    frontal_channels: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
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
            psd_res = compute_psd_bands(raw_data, fs=fs, frontal_channels=frontal_channels)
        else:
            psd_res = {
                "theta_global": np.nan, "alpha_global": np.nan, "beta_global": np.nan,
                "theta_fm": np.nan, "eeg_duration_s": 0.0, "eeg_reliability": "insufficient",
                "artifact_fraction": 0.0, "log_theta": np.nan, "log_alpha": np.nan, "log_beta": np.nan
            }

        alpha_val = psd_res["alpha_global"]
        if np.isfinite(alpha_val):
            alpha_values.append(alpha_val)

        parsed.append({
            "sentence_index": sentence_index,
            "sentence_text": str(content) if content is not None else "",
            "sentence_obj": sentence,
            "psd": psd_res,
        })

    baseline_alpha = float(np.median(alpha_values)) if alpha_values else np.nan
    records = []

    for item in parsed:
        psd = item["psd"]
        theta_glob = psd["theta_global"]
        alpha_glob = psd["alpha_global"]
        beta_glob = psd["beta_global"]
        theta_fm = psd["theta_fm"]

        eei = (
            beta_glob / (alpha_glob + theta_glob + EPSILON)
            if np.isfinite(beta_glob) and np.isfinite(alpha_glob) and np.isfinite(theta_glob)
            else np.nan
        )
        # FMT defaults to true frontal midline theta (theta_fm); fallback to theta_global
        fmt = theta_fm if np.isfinite(theta_fm) else theta_glob
        asm = (
            (baseline_alpha - alpha_glob) / (baseline_alpha + EPSILON)
            if np.isfinite(baseline_alpha) and np.isfinite(alpha_glob)
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
            "ThetaPower": theta_glob,
            "AlphaPower": alpha_glob,
            "BetaPower": beta_glob,
            "theta_global": theta_glob,
            "theta_fm": theta_fm,
            "eeg_duration_s": psd["eeg_duration_s"],
            "eeg_reliability": psd["eeg_reliability"],
            "artifact_fraction": psd["artifact_fraction"],
            "log_theta": psd["log_theta"],
            "log_alpha": psd["log_alpha"],
            "log_beta": psd["log_beta"],
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
