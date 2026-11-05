"""
NREI/eye-tracking feature extraction for ZuCo MATLAB files.

This preserves the formulas in the user's supplied extractor:
    EEI = BetaPower / (AlphaPower + ThetaPower + epsilon)
    FMT = whole-scalp ThetaPower proxy
    ASM = (participant median AlphaPower - sentence AlphaPower)
          / (participant median AlphaPower + epsilon)

Also extracts sentence-level eye-tracking features:
    GD, TRT, FFD, meanPupilSize, nFixations

Important:
- This code does NOT invent a single composite "NREI score"; the supplied
  method defines EEI, FMT and ASM as separate EEG-derived metrics.
- The supplied code assumes rawData is channels x time and fs=500 Hz.
  Check this against the exact ZuCo files/metadata before interpreting results.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch

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


def compute_psd_bands(eeg_signal, fs=500):
    """
    Calculate mean channel-wise absolute band power using Welch PSD.
    Input must be (channels, samples). Returns NaN triplet if unusable.
    """
    try:
        signal = np.asarray(eeg_signal, dtype=float)
        signal = np.squeeze(signal)

        if signal.ndim != 2:
            return np.nan, np.nan, np.nan

        # Use the longer dimension as time if data appears transposed.
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        if signal.shape[0] < 1 or signal.shape[1] < fs:
            return np.nan, np.nan, np.nan

        signal = np.where(np.isfinite(signal), signal, np.nan)
        # Welch cannot accept NaNs. Drop channels containing non-finite samples.
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
            # Integrate each channel's PSD over frequency, then average channels.
            return float(np.mean(np.trapz(psd[:, mask], freqs[mask], axis=-1)))

        return band_power(4.0, 8.0), band_power(8.0, 13.0), band_power(13.0, 30.0)

    except Exception:
        return np.nan, np.nan, np.nan


def extract_patient_nrei(mat_path, fs=500):
    """Extract sentence-level features from one participant's .mat file."""
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

    # Participant-specific baseline is calculated from this participant's
    # sentences only. It does not use the held-out patient's labels.
    baseline_alpha = (
        float(np.median(alpha_values)) if alpha_values else np.nan
    )

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

        # FMT is a whole-scalp theta-power proxy in this implementation,
        # not channel-isolated frontal midline theta.
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


def extract_folder(mat_folder, output_csv, fs=500, exclude_paths=None):
    """Extract all .mat files in a folder, optionally excluding held-out files."""
    mat_folder = Path(mat_folder)
    exclude = {str(Path(p).resolve()) for p in (exclude_paths or [])}
    mat_files = sorted(
        [p for p in mat_folder.glob("*.mat") if str(p.resolve()) not in exclude],
        key=lambda p: p.name.lower()
    )

    if not mat_files:
        raise FileNotFoundError(f"No eligible .mat files found in {mat_folder}")

    all_records = []
    for mat_file in mat_files:
        patient_id = mat_file.stem
        print(f"[EXTRACT] {mat_file.name}")
        records = extract_patient_nrei(mat_file, fs=fs)

        for record in records:
            record["patient_id"] = patient_id
            record["source_file"] = mat_file.name
            all_records.append(record)

    if not all_records:
        raise RuntimeError("No sentence records were extracted from the .mat files.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(output_csv, index=False)
    print(f"[SAVED] {output_csv} ({len(all_records)} sentence rows)")
    return pd.DataFrame(all_records)
