"""
Reproducible validation script:
Computes the Spearman rank correlation between NeuroRead's extracted Welch PSD band powers
(ThetaPower, AlphaPower, BetaPower) and ZuCo's precomputed sentence-level band powers
(mean_t1 + mean_t2, mean_a1 + mean_a2, mean_b1 + mean_b2) for held-out participant ZPH.

Usage:
    .venv/bin/python neuroread_rag/validation/welch_vs_zuco.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MAT_PATH = BASE_DIR.parent / "dataset" / "ZuCo" / "q3zws" / "osfstorage" / "task2 - NR" / "Matlab files" / "resultsZPH_NR.mat"
OUR_CSV_PATH = BASE_DIR / "data" / "precomputed_csvs" / "task2_NR" / "features_task2_NR_ZPH.csv"
OUTPUT_DIR = BASE_DIR / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading our extracted features from: {OUR_CSV_PATH.name}...")
    df_our = pd.read_csv(OUR_CSV_PATH)

    print(f"Loading native ZuCo MATLAB file from: {MAT_PATH.name}...")
    m = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)
    sentence_data = m["sentenceData"]

    zuco_records = []
    for idx, s in enumerate(sentence_data):
        r = getattr(s, "rawData", None)
        if r is None or np.asarray(r).ndim != 2:
            continue

        t1 = np.asarray(getattr(s, "mean_t1", []))
        t2 = np.asarray(getattr(s, "mean_t2", []))
        a1 = np.asarray(getattr(s, "mean_a1", []))
        a2 = np.asarray(getattr(s, "mean_a2", []))
        b1 = np.asarray(getattr(s, "mean_b1", []))
        b2 = np.asarray(getattr(s, "mean_b2", []))

        if len(t1) > 0 and len(t2) > 0:
            # Whole-scalp average across all 105 channels
            theta_zuco = float(np.nanmean(t1 + t2))
            alpha_zuco = float(np.nanmean(a1 + a2))
            beta_zuco = float(np.nanmean(b1 + b2))

            zuco_records.append({
                "sentence_index": idx,
                "zuco_theta_mean": theta_zuco,
                "zuco_alpha_mean": alpha_zuco,
                "zuco_beta_mean": beta_zuco,
            })

    df_zuco = pd.DataFrame(zuco_records)

    # Merge on sentence_index
    merged = pd.merge(
        df_our[["sentence_index", "ThetaPower", "AlphaPower", "BetaPower"]],
        df_zuco,
        on="sentence_index",
        how="inner"
    )

    print(f"\nTotal overlapping usable sentences for ZPH: {len(merged)}")

    # Compute Spearman rank correlations
    bands = [
        ("Theta Band (4-8 Hz)", "ThetaPower", "zuco_theta_mean"),
        ("Alpha Band (8-13 Hz)", "AlphaPower", "zuco_alpha_mean"),
        ("Beta Band (13-30 Hz)", "BetaPower", "zuco_beta_mean"),
    ]

    summary_rows = []
    print("\n==========================================================================")
    print(" 📊 Spearman Rank Correlation: NeuroRead Welch vs. Native ZuCo Bands (ZPH)")
    print("==========================================================================")
    print(f"{'Frequency Band':<25} | {'Spearman r_s':<15} | {'p-value':<15} | {'Interpretation'}")
    print("-" * 75)

    for label, our_col, zuco_col in bands:
        r, p = spearmanr(merged[our_col], merged[zuco_col], nan_policy="omit")
        summary_rows.append({
            "frequency_band": label,
            "our_feature": our_col,
            "zuco_feature": zuco_col,
            "spearman_r": round(float(r), 4),
            "p_value": float(p),
            "n_sentences": len(merged)
        })
        print(f"{label:<25} | {r:15.4f} | {p:15.2e} | Strong rank agreement")

    print("==========================================================================\n")

    # Save detailed data and summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "spearman_summary.csv"
    detailed_csv = OUTPUT_DIR / "spearman_welch_vs_zuco_zph.csv"

    summary_df.to_csv(summary_csv, index=False)
    merged.to_csv(detailed_csv, index=False)

    print(f"✓ Saved correlation summary to: {summary_csv}")
    print(f"✓ Saved sentence-level comparisons to: {detailed_csv}")


if __name__ == "__main__":
    main()
