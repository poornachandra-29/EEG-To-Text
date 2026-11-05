"""
visualise.py
------------
Visualise the dataset and GMM clustering results.

Usage:
    python visualise.py
    python visualise.py --input predicted_states.csv
    python visualise.py --input output_cluster_labels.csv
    python visualise.py --input all_patients_advanced_nrei.csv

The script creates a folder named "visualisations" containing PNG charts
and a text summary. It uses matplotlib only (no seaborn).

It can visualise:
- Total rows, patients, features, missing values
- Rows per patient
- Cluster counts and percentages
- Cluster distribution within each patient
- Feature distributions by cluster
- Cluster feature profiles (mean values)
- Feature correlation matrix
- PCA projection of observations
- GMM membership-confidence distribution, if probability columns exist
"""

import argparse
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount"
]


def choose_input_file():
    """Choose the most useful available CSV when --input is not supplied."""
    candidates = [
        "predicted_states.csv",
        "output_cluster_labels.csv",
        "cluster_labels.csv",
        "preprocessed_features.csv",
        "all_patients_advanced_nrei.csv",
    ]
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    raise FileNotFoundError(
        "No CSV found. Put your CSV in this folder or use --input filename.csv"
    )


def save_plot(filename, output_dir):
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def get_feature_columns(df):
    features = [col for col in DEFAULT_FEATURES if col in df.columns]
    if features:
        return features

    excluded = {
        "patient_id", "cluster_id", "cluster_name", "cognitive_state",
        "prediction_status", "cluster_probability"
    }
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    return [col for col in numeric if col not in excluded
            and not col.startswith("prob_cluster_")]


def write_summary(df, features, cluster_col, output_dir, input_file):
    summary_path = os.path.join(output_dir, "dataset_summary.txt")
    lines = [
        f"INPUT FILE: {input_file}",
        f"TOTAL ROWS / OBSERVATIONS: {len(df):,}",
        f"TOTAL COLUMNS: {len(df.columns):,}",
    ]

    if "patient_id" in df.columns:
        lines.append(f"NUMBER OF PATIENTS: {df['patient_id'].nunique():,}")
        lines.append("\nROWS PER PATIENT:")
        lines.append(df["patient_id"].value_counts().sort_index().to_string())
    else:
        lines.append("PATIENT ID COLUMN: not found")

    lines.append(f"\nFEATURES USED FOR VISUALISATION: {features}")
    lines.append("\nMISSING VALUES PER COLUMN:")
    lines.append(df.isna().sum().to_string())

    if cluster_col:
        lines.append(f"\nCLUSTER COLUMN: {cluster_col}")
        lines.append("\nCLUSTER COUNTS:")
        lines.append(df[cluster_col].value_counts(dropna=False).to_string())
        lines.append("\nCLUSTER PERCENTAGES:")
        lines.append(
            (df[cluster_col].value_counts(normalize=True, dropna=False) * 100)
            .round(2).astype(str).add("%").to_string()
        )
    else:
        lines.append("\nNO CLUSTER COLUMN FOUND.")
        lines.append(
            "This file contains observations/features but no cluster assignments. "
            "Run train_clustering.py or predict.py first for cluster visualisations."
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:", summary_path)


def plot_dataset_overview(df, features, output_dir):
    # Rows per patient
    if "patient_id" in df.columns:
        counts = df["patient_id"].value_counts().sort_index()
        plt.figure(figsize=(11, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.title("Number of Observations per Patient")
        plt.xlabel("Patient ID")
        plt.ylabel("Number of observations")
        plt.xticks(rotation=45, ha="right")
        save_plot("01_observations_per_patient.png", output_dir)

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        plt.figure(figsize=(11, 5))
        plt.bar(missing.index.astype(str), missing.values)
        plt.title("Missing Values per Column")
        plt.xlabel("Column")
        plt.ylabel("Missing value count")
        plt.xticks(rotation=60, ha="right")
        save_plot("02_missing_values.png", output_dir)

    # Feature distributions
    valid_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    if valid_features:
        n = len(valid_features)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.5 * n)))
        if n == 1:
            axes = [axes]
        for ax, feature in zip(axes, valid_features):
            values = pd.to_numeric(df[feature], errors="coerce").dropna()
            ax.hist(values, bins=35, edgecolor="black")
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
        save_plot("03_feature_distributions.png", output_dir)


def plot_cluster_counts(df, cluster_col, output_dir):
    if not cluster_col:
        return

    counts = df[cluster_col].fillna("Missing").astype(str).value_counts()
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(10, 5))
    bars = plt.bar(counts.index, counts.values, edgecolor="black")
    plt.title("Number of Observations in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of observations")
    plt.xticks(rotation=30, ha="right")

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9
        )
    save_plot("04_cluster_counts.png", output_dir)

    plt.figure(figsize=(8, 8))
    plt.pie(
        counts.values,
        labels=[f"{name}\n{pct:.1f}%" for name, pct in zip(counts.index, percentages)],
        startangle=90,
        autopct=None
    )
    plt.title("Percentage of Observations in Each Cluster")
    save_plot("05_cluster_percentages.png", output_dir)


def plot_patient_cluster_distribution(df, cluster_col, output_dir):
    if not cluster_col or "patient_id" not in df.columns:
        return

    table = pd.crosstab(df["patient_id"], df[cluster_col])
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Cluster Distribution for Each Patient")
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Number of observations")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    save_plot("06_patient_cluster_distribution.png", output_dir)

    # Within-patient percentages make patients with different row counts comparable.
    pct_table = table.div(table.sum(axis=1), axis=0) * 100
    ax = pct_table.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title("Within-Patient Cluster Percentages")
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Percentage of patient's observations")
    ax.set_ylim(0, 100)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    save_plot("07_patient_cluster_percentages.png", output_dir)


def plot_features_by_cluster(df, features, cluster_col, output_dir):
    if not cluster_col:
        return

    valid_features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    for feature in valid_features:
        groups, labels = [], []
        for label, group in df.groupby(cluster_col, dropna=False):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if len(values):
                groups.append(values.values)
                labels.append(str(label))

        if not groups:
            continue

        plt.figure(figsize=(10, 5))
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title(f"{feature} Distribution by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        plt.xticks(rotation=25, ha="right")
        safe_name = "".join(c if c.isalnum() else "_" for c in feature)
        save_plot(f"feature_{safe_name}_by_cluster.png", output_dir)

    # Mean feature profile, useful for comparing cluster patterns.
    means = df.groupby(cluster_col)[valid_features].mean(numeric_only=True)
    if not means.empty:
        ax = means.T.plot(kind="bar", figsize=(13, 6))
        ax.set_title("Mean Feature Values for Each Cluster")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean feature value")
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        save_plot("cluster_mean_feature_profiles.png", output_dir)

        means.to_csv(os.path.join(output_dir, "cluster_feature_means.csv"))


def plot_correlation(df, features, output_dir):
    valid = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]
    if len(valid) < 2:
        return

    corr = df[valid].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(valid)))
    ax.set_yticks(range(len(valid)))
    ax.set_xticklabels(valid, rotation=45, ha="right")
    ax.set_yticklabels(valid)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    save_plot("08_feature_correlation.png", output_dir)


def plot_pca(df, features, cluster_col, output_dir):
    valid = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]
    if len(valid) < 2:
        return

    work = df[valid].replace([np.inf, -np.inf], np.nan)
    keep = work.notna().all(axis=1)
    work = work.loc[keep]

    if len(work) < 3:
        return

    scaled = StandardScaler().fit_transform(work)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)

    plt.figure(figsize=(9, 7))

    if cluster_col and cluster_col in df.columns:
        labels = df.loc[keep, cluster_col].astype(str)
        for label in sorted(labels.unique()):
            mask = labels.values == label
            plt.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=18,
                alpha=0.65,
                label=label
            )
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        title = "PCA Projection of Observations by Cluster"
    else:
        plt.scatter(coordinates[:, 0], coordinates[:, 1], s=18, alpha=0.65)
        title = "PCA Projection of Observations"

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    save_plot("09_pca_cluster_visualisation.png", output_dir)


def plot_probabilities(df, output_dir):
    probability_cols = [
        col for col in df.columns if col.startswith("prob_cluster_")
    ]

    if probability_cols:
        values = df[probability_cols].max(axis=1).dropna()
    elif "cluster_probability" in df.columns:
        values = pd.to_numeric(
            df["cluster_probability"], errors="coerce"
        ).dropna()
    else:
        return

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=30, edgecolor="black")
    plt.title("Distribution of Maximum GMM Membership Probability")
    plt.xlabel("Maximum membership probability")
    plt.ylabel("Number of observations")
    save_plot("10_cluster_membership_confidence.png", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise dataset and clustering results."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV. If omitted, the script searches common output filenames."
    )
    parser.add_argument(
        "--output-dir",
        default="visualisations",
        help="Folder in which charts and summary are saved."
    )
    args = parser.parse_args()

    input_file = args.input or choose_input_file()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Reading:", input_file)
    df = pd.read_csv(input_file)

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    features = get_feature_columns(df)

    # Prefer readable cluster names if present; otherwise use cluster IDs.
    if "cluster_name" in df.columns:
        cluster_col = "cluster_name"
    elif "cognitive_state" in df.columns:
        cluster_col = "cognitive_state"
    elif "cluster_id" in df.columns:
        cluster_col = "cluster_id"
    else:
        cluster_col = None

    print("\n========== DATASET OVERVIEW ==========")
    print("Input file:", input_file)
    print("Rows / observations:", f"{len(df):,}")
    print("Columns:", len(df.columns))
    print("Patients:", df["patient_id"].nunique() if "patient_id" in df else "Not available")
    print("Features:", features)
    print("Cluster column:", cluster_col or "Not found")

    if cluster_col:
        print("\nCluster counts:")
        print(df[cluster_col].value_counts(dropna=False))
        print("\nCluster percentages:")
        print(
            (df[cluster_col].value_counts(normalize=True, dropna=False) * 100)
            .round(2).astype(str).add("%")
        )

    write_summary(df, features, cluster_col, output_dir, input_file)
    plot_dataset_overview(df, features, output_dir)
    plot_cluster_counts(df, cluster_col, output_dir)
    plot_patient_cluster_distribution(df, cluster_col, output_dir)
    plot_features_by_cluster(df, features, cluster_col, output_dir)
    plot_correlation(df, features, output_dir)
    plot_pca(df, features, cluster_col, output_dir)
    plot_probabilities(df, output_dir)

    print("\nVisualisation completed.")
    print("All charts and summary are in:", output_dir)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
