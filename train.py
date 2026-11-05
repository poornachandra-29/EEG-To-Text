"""
Train the unsupervised GMM using all but the last .mat file.

The held-out patient is selected deterministically by sorting .mat filenames
alphabetically and reserving the final filename. Review train_manifest.json
to confirm this is the patient you intend to hold out.

Usage:
    python train.py --mat-root "/path/to/Matlab files"

Optional:
    python train.py --mat-root "/path/to/Matlab files" --min-k 2 --max-k 6 --fs 500
"""


import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from nrei_extractor import extract_folder, MODEL_FEATURES
from preprocess import fit_preprocessor


# ================= USER SETTINGS =================

MAT_ROOT = r"/mnt/DATA/SPLAB/1. Code/EEG-To-Text/dataset/ZuCo/q3zws/osfstorage/task2 - NR/Matlab files/"

OUTPUT_DIR = "artifacts"

MIN_K = 2
MAX_K = 6

SAMPLING_FREQUENCY = 500

# ==================================================


def main():
    root = Path(MAT_ROOT).expanduser().resolve()
    output = Path(OUTPUT_DIR).resolve()

    min_k = MIN_K
    max_k = MAX_K
    fs = SAMPLING_FREQUENCY

    output.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise NotADirectoryError(f"MAT_ROOT does not exist: {root}")

    mat_files = sorted(
        root.glob("*.mat"),
        key=lambda p: p.name.lower()
    )

    if len(mat_files) != 12:
        print(f"WARNING: Expected 12 files, found {len(mat_files)}")

    if len(mat_files) < 3:
        raise ValueError("Not enough MATLAB files.")

    # Reserve the last alphabetically sorted file for testing
    train_files = mat_files[:-1]
    test_file = mat_files[-1]

    print("\n========== PATIENT SPLIT ==========")
    print("Training patients:", len(train_files))

    for file in train_files:
        print("TRAIN:", file.name)

    print("HELD-OUT TEST:", test_file.name)

    # Save split information
    manifest = {
        "mat_root": str(root),
        "train_files": [str(p.resolve()) for p in train_files],
        "heldout_test_file": str(test_file.resolve()),
        "sampling_frequency_hz": fs,
        "model_features": MODEL_FEATURES
    }

    (output / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Extract NREI features from training patients only
    train_raw = extract_folder(
        root,
        output / "train_nrei_raw.csv",
        fs=fs,
        exclude_paths=[test_file]
    )

    # Preprocess training data
    train_df, x_train = fit_preprocessor(train_raw, output)

    train_df.to_csv(
        output / "train_nrei_preprocessed.csv",
        index=False
    )

    # Select optimal K using BIC
    results = []
    best_model = None
    best_bic = float("inf")
    best_k = None

    for k in range(min_k, max_k + 1):

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=10,
            reg_covar=1e-5,
            random_state=42
        )

        model.fit(x_train)

        bic = model.bic(x_train)
        aic = model.aic(x_train)

        results.append({
            "n_components": k,
            "BIC": bic,
            "AIC": aic
        })

        print(f"K={k} | BIC={bic:.3f} | AIC={aic:.3f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = model

    if best_model is None:
        raise RuntimeError("GMM training failed.")

    pd.DataFrame(results).to_csv(
        output / "gmm_model_selection.csv",
        index=False
    )

    joblib.dump(
        best_model,
        output / "gmm_model.joblib"
    )

    # Generate training cluster assignments
    clusters = best_model.predict(x_train)
    probabilities = best_model.predict_proba(x_train)

    train_output = train_df.copy()

    train_output["cluster_id"] = clusters
    train_output["cluster_name"] = [
        f"Cluster {i}" for i in clusters
    ]
    train_output["cluster_probability"] = probabilities.max(axis=1)

    for i in range(best_k):
        train_output[f"prob_cluster_{i}"] = probabilities[:, i]

    train_output.to_csv(
        output / "train_cluster_assignments.csv",
        index=False
    )

    # Generate cluster profiles
    profiles = train_output.groupby("cluster_id")[MODEL_FEATURES].agg(
        ["count", "mean", "median", "std"]
    )

    profiles.to_csv(
        output / "train_cluster_profiles.csv"
    )

    # Save model metadata
    metadata = {
        "n_components": int(best_k),
        "features": MODEL_FEATURES,
        "selection_method": "BIC",
        "best_bic": float(best_bic),
        "sampling_frequency_hz": fs
    }

    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n========== TRAINING COMPLETE ==========")
    print("Selected clusters:", best_k)
    print("Training BIC:", best_bic)
    print("Held-out patient:", test_file.name)
    print("Artifacts saved in:", output)


if __name__ == "__main__":
    main()
