"""
Evaluate/assign the held-out patient using the trained training-only pipeline.

Usage:
    python test.py --mat-file "/path/to/resultsZPH_NR.mat"

This script:
1. Extracts the same sentence-level NREI/eye-tracking features from the test .mat.
2. Saves raw features under test_nrei_csv/.
3. Applies training-fitted medians and RobustScaler (does not refit).
4. Predicts GMM cluster assignments and membership probabilities.
5. Saves test features and predictions under test_nrei_csv/.
"""


import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nrei_extractor import extract_patient_nrei, MODEL_FEATURES
from preprocess import transform_with_saved_preprocessor

# ================= USER SETTINGS =================

TEST_MAT_FILE = r"/mnt/DATA/SPLAB/1. Code/EEG-To-Text/dataset/ZuCo/q3zws/osfstorage/task2 - NR/Matlab files/resultsZPH_NR.mat"

ARTIFACT_DIR = "artifacts"

OUTPUT_DIR = "test_nrei_csv"

CONFIDENCE_THRESHOLD = 0.55

# ==================================================

def main():
    mat_file = Path(TEST_MAT_FILE).expanduser().resolve()
    artifact_dir = Path(ARTIFACT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not mat_file.is_file():
        raise FileNotFoundError(
            f"Test .mat file not found: {mat_file}"
        )

    manifest_path = artifact_dir / "train_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Training artifacts not found. Run train.py first."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    train_paths = {
        str(Path(p).resolve())
        for p in manifest["train_files"]
    }

    if str(mat_file) in train_paths:
        raise ValueError(
            "This patient was already used for training!"
        )

    expected_test = str(
        Path(manifest["heldout_test_file"]).resolve()
    )

    if str(mat_file) != expected_test:
        raise ValueError(
            f"Expected held-out file: {expected_test}\n"
            f"Received: {mat_file}"
        )

    fs = float(manifest["sampling_frequency_hz"])

    # 1. Extract NREI features
    records = extract_patient_nrei(mat_file, fs=fs)

    if not records:
        raise RuntimeError("No features extracted.")

    raw_df = pd.DataFrame(records)

    raw_df["patient_id"] = mat_file.stem
    raw_df["source_file"] = mat_file.name

    raw_path = output_dir / "test_nrei_raw.csv"

    raw_df.to_csv(raw_path, index=False)

    print("[SAVED]", raw_path)

    # 2. Apply training preprocessing
    test_df, x_test = transform_with_saved_preprocessor(
        raw_df,
        artifact_dir
    )

    preprocessed_path = (
        output_dir / "test_nrei_preprocessed.csv"
    )

    test_df.to_csv(preprocessed_path, index=False)

    print("[SAVED]", preprocessed_path)

    # 3. Load trained GMM
    model = joblib.load(
        artifact_dir / "gmm_model.joblib"
    )

    # 4. Predict clusters
    cluster_ids = model.predict(x_test)

    probabilities = model.predict_proba(x_test)

    max_prob = probabilities.max(axis=1)

    result = test_df.copy()

    result["cluster_id"] = cluster_ids

    result["cluster_name"] = [
        f"Cluster {i}" for i in cluster_ids
    ]

    result["cluster_probability"] = max_prob

    result["prediction_status"] = np.where(
        max_prob >= CONFIDENCE_THRESHOLD,
        "Higher cluster membership confidence",
        "Uncertain - review"
    )

    for i in range(model.n_components):
        result[f"prob_cluster_{i}"] = probabilities[:, i]

    # 5. Save predictions
    prediction_path = (
        output_dir / "test_predictions.csv"
    )

    result.to_csv(prediction_path, index=False)

    # 6. Generate summary
    counts = (
        result["cluster_name"]
        .value_counts()
        .sort_index()
    )

    summary = {
        "test_file": str(mat_file),
        "patient_id": mat_file.stem,
        "sentence_rows_extracted": int(len(result)),
        "cluster_counts": {
            str(k): int(v) for k, v in counts.items()
        },
        "mean_max_membership_probability": float(
            max_prob.mean()
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "note": (
            "GMM membership probability is not a probability "
            "of concentration or a validated cognitive state."
        )
    }

    summary_path = output_dir / "test_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== TESTING COMPLETE ==========")

    print("Patient:", mat_file.stem)
    print("Sentence rows:", len(result))

    print("\nPredicted cluster counts:")
    print(counts)

    print(
        "\nMean membership confidence:",
        round(float(max_prob.mean()), 4)
    )

    print("\nAll files saved in:", output_dir)


if __name__ == "__main__":
    main()
