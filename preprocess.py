"""
Preprocessing shared by train.py and test.py.

The imputation medians and RobustScaler are fitted ONLY on training data.
The exact saved preprocessing parameters are reused for the held-out patient.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from nrei_extractor import MODEL_FEATURES


def clean_feature_table(df):
    """Keep metadata and model features; coerce invalid numeric values to NaN."""
    required = MODEL_FEATURES + ["patient_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    for col in MODEL_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows only if every model feature is missing.
    result = result.loc[result[MODEL_FEATURES].notna().any(axis=1)].copy()
    if result.empty:
        raise ValueError("No usable rows: all model features are missing.")
    return result


def fit_preprocessor(train_df, artifact_dir):
    """Fit training medians and scaler; save them for test-time reuse."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = clean_feature_table(train_df)
    medians = train_df[MODEL_FEATURES].median()
    # Entirely-missing training feature: use 0 as an explicit fallback.
    medians = medians.fillna(0.0)

    x_filled = train_df[MODEL_FEATURES].fillna(medians)
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_filled)

    joblib.dump(medians.to_dict(), artifact_dir / "feature_medians.joblib")
    joblib.dump(scaler, artifact_dir / "robust_scaler.joblib")

    return train_df, x_scaled


def transform_with_saved_preprocessor(df, artifact_dir):
    """Apply training medians and scaler without refitting on test data."""
    artifact_dir = Path(artifact_dir)
    medians = joblib.load(artifact_dir / "feature_medians.joblib")
    scaler = joblib.load(artifact_dir / "robust_scaler.joblib")

    df = clean_feature_table(df)
    x_filled = df[MODEL_FEATURES].fillna(pd.Series(medians))
    x_scaled = scaler.transform(x_filled)
    return df, x_scaled
