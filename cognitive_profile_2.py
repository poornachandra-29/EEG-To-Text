#!/usr/bin/env python3
"""
ZuCo Cognitive Profile Analysis

Input: sentence-level CSV from the ZuCo feature extraction pipeline.

Outputs in OUTPUT_DIR:
  cognitive_profile_output.csv
  cognitive_profile_summary.txt
  personal_reference_ranges.csv
  cluster_feature_analysis.csv
  cognitive_profile_timeline.png
  cognitive_state_timeline.png
  cluster_distribution.png
  cognitive_state_predictions.csv
  cluster_cognitive_interpretation.csv

Install:
  pip install pandas numpy matplotlib

Run:
  1) Edit INPUT_CSV below.
  2) python3 cognitive_profile.py

Scientific cautions:
- These composite indices are exploratory, not validated clinical scales.
- There are no universal optimal values for these measures in this CSV.
- Personal reference ranges describe this session, not healthy/optimal values.
- Cognitive labels below are conservative, literature-informed heuristic pattern labels,
  not validated diagnoses or ground-truth mental states.
- Cluster IDs are arbitrary; cluster interpretations are derived from the supplied
  participant CSV and must not be treated as permanent cross-participant labels.
- cluster_probability is GMM membership strength, not probability of concentration.
- Sentence index is sequence order, not elapsed clock time.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- SETTINGS ---------------------------
INPUT_CSV = r"/mnt/DATA/SPLAB/1. Code/EEG-To-Text/test_nrei_csv/test_predictions.csv"  # CHANGE THIS
OUTPUT_DIR = r"cognitive_profile_results_type_2"
ROLLING_WINDOW = 10
MIN_PERIODS = 3
# Heuristic evidence thresholds in robust-Z units; NOT clinical cutoffs.
EVIDENCE_THRESHOLD = 0.50
STRONG_EVIDENCE_THRESHOLD = 1.00
AMBIGUITY_MARGIN = 0.25
REFERENCE_MAD_MULTIPLIER = 1.5
EPS = 1e-9
# ---------------------------------------------------------------

FEATURES = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD", "PupilSize",
    "FixationCount", "ThetaPower", "AlphaPower", "BetaPower"
]

INDEX_INFO = [
    ("SAI", "Engagement-Stability Evidence Proxy",
     "SAI_percentile", "SAI_category"),
    ("REI", "Engagement-Related Signal Index",
     "REI_percentile", "REI_category"),
    ("MWLI_proxy", "Attention-Instability Evidence Proxy",
     "MWLI_percentile", "MWLI_category"),
    ("CWI", "Cognitive Workload Evidence Index",
     "CWI_percentile", "CWI_category"),
    ("VSI_proxy", "Signal-Stability Proxy (not a vigilance measure)",
     "VSI_percentile", "VSI_category"),
]


def robust_z(series):
    """Robust within-participant Z-score based on median and MAD."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = x.median()
    mad = (x - med).abs().median()

    if pd.isna(med):
        return pd.Series(np.nan, index=x.index, dtype=float)

    if pd.isna(mad) or mad < EPS:
        sd = x.std()
        if pd.isna(sd) or sd < EPS:
            return pd.Series(np.where(x.notna(), 0.0, np.nan),
                             index=x.index, dtype=float)
        return (x - x.mean()) / sd

    return 0.67448975 * (x - med) / mad


def row_mean(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].mean(axis=1, skipna=True)


def prepare_data(df):
    if "patient_id" not in df.columns:
        df["patient_id"] = "participant_1"

    if "sentence_index" not in df.columns:
        df["sentence_index"] = df.groupby("patient_id").cumcount()

    df["sentence_index"] = pd.to_numeric(
        df["sentence_index"], errors="coerce"
    )

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(
        ["patient_id", "sentence_index"], kind="stable"
    ).reset_index(drop=True)

    if not any(c in df.columns for c in FEATURES):
        raise ValueError(
            "No expected ZuCo feature columns found. Check the CSV header."
        )
    return df


def add_z_features(df):
    zcols = FEATURES.copy()

    # Derive ratios only if their source columns exist.
    if {"ThetaPower", "AlphaPower"}.issubset(df.columns):
        df["Theta_Alpha_Ratio"] = (
            df["ThetaPower"] / (df["AlphaPower"].abs() + EPS)
        )
        zcols.append("Theta_Alpha_Ratio")

    if {"ThetaPower", "BetaPower"}.issubset(df.columns):
        df["Theta_Beta_Ratio"] = (
            df["ThetaPower"] / (df["BetaPower"].abs() + EPS)
        )
        zcols.append("Theta_Beta_Ratio")

    for col in zcols:
        if col in df.columns:
            df[col + "_rz"] = (
                df.groupby("patient_id", sort=False)[col]
                .transform(robust_z)
            )
    return df


def add_variability(df):
    pairs = [
        ("EEI", "EEI_local_sd"),
        ("FFD", "FFD_local_sd"),
        ("TRT", "TRT_local_sd"),
        ("PupilSize", "Pupil_local_sd"),
    ]

    for source, target in pairs:
        if source in df.columns:
            df[target] = (
                df.groupby("patient_id", sort=False)[source]
                .transform(
                    lambda s: s.rolling(
                        ROLLING_WINDOW, min_periods=MIN_PERIODS
                    ).std()
                )
            )
            df[target + "_rz"] = (
                df.groupby("patient_id", sort=False)[target]
                .transform(robust_z)
            )
    return df


def add_indices(df):
    """Create five transparent, exploratory composite indices."""
    df = add_z_features(df)

    # REI: engagement-related EEG/pupil evidence; not concentration itself.
    df["REI"] = row_mean(
        df, ["EEI_rz", "FMT_rz", "PupilSize_rz"]
    )

    # CWI: workload-related evidence from theta, pupil and reading times.
    # FMT and ThetaPower are identical in the supplied feature CSV; count theta once.
    theta_col = "ThetaPower_rz" if "ThetaPower_rz" in df.columns else "FMT_rz"
    df["CWI"] = row_mean(
        df, [theta_col, "PupilSize_rz", "GD_rz", "TRT_rz"]
    )

    df = add_variability(df)

    variability_cols = [
        c for c in [
            "EEI_local_sd_rz", "FFD_local_sd_rz",
            "TRT_local_sd_rz", "Pupil_local_sd_rz"
        ] if c in df.columns
    ]
    instability = row_mean(df, variability_cols)

    # SAI: relative engagement evidence minus local signal/reading variability.
    df["SAI"] = df["REI"] - instability

    # MWLI proxy: inverse EEI plus local reading variability.
    reading_var = row_mean(
        df, ["FFD_local_sd_rz", "TRT_local_sd_rz"]
    )
    if "EEI_rz" in df.columns:
        df["MWLI_proxy"] = -df["EEI_rz"] + reading_var
    else:
        df["MWLI_proxy"] = np.nan

    # VSI: stability of available signals, not validated vigilance/fatigue.
    df["VSI_proxy"] = -instability

    # Percentile rank within each participant's own session.
    for raw, pct in [
        ("SAI", "SAI_percentile"),
        ("REI", "REI_percentile"),
        ("MWLI_proxy", "MWLI_percentile"),
        ("CWI", "CWI_percentile"),
        ("VSI_proxy", "VSI_percentile"),
    ]:
        df[pct] = (
            df.groupby("patient_id", sort=False)[raw]
            .rank(pct=True, method="average") * 100
        )

    # Relative tertile categories, not validated state labels.
    category_specs = [
        ("SAI", "SAI_category",
         "Lower relative stability", "Moderate relative stability",
         "Higher relative stability"),
        ("REI", "REI_category",
         "Lower engagement-related evidence",
         "Moderate engagement-related evidence",
         "Higher engagement-related evidence"),
        ("MWLI_proxy", "MWLI_category",
         "Lower drift-like evidence", "Moderate drift-like evidence",
         "Higher drift-like evidence"),
        ("CWI", "CWI_category",
         "Lower workload-related evidence",
         "Moderate workload-related evidence",
         "Higher workload-related evidence"),
        ("VSI_proxy", "VSI_category",
         "Lower signal stability", "Moderate signal stability",
         "Higher signal stability"),
    ]

    for source, target, low, middle, high in category_specs:
        def make_labels(s):
            valid = s.dropna()
            if len(valid) < 3:
                return pd.Series("Insufficient data", index=s.index)
            q1, q2 = valid.quantile([1 / 3, 2 / 3])
            labels = np.select(
                [s <= q1, s >= q2],
                [low, high],
                default=middle
            )
            return pd.Series(labels, index=s.index).where(
                s.notna(), "Insufficient data"
            )

        df[target] = (
            df.groupby("patient_id", sort=False)[source]
            .transform(make_labels)
        )

    return df


def add_cognitive_evidence_scores(df):
    """Add transparent, provisional pattern-evidence scores.

    Scores are means of participant-relative robust z features. They are NOT
    probabilities and the thresholds are engineering heuristics, not validated
    physiological cutoffs.
    """
    # Avoid double-counting FMT: in the supplied CSV FMT == ThetaPower exactly.
    # EEI already contains beta/(alpha+theta), so BetaPower is not added again.
    negative_sources = [
        ("EEI_rz", "low_EEI_evidence"),
        ("FFD_local_sd_rz", "negative_FFD_variability"),
        ("TRT_local_sd_rz", "negative_TRT_variability"),
        ("EEI_local_sd_rz", "negative_EEI_variability"),
    ]
    for source, target in negative_sources:
        if source in df.columns:
            df[target] = -pd.to_numeric(df[source], errors="coerce")

    # Engagement-supported + stable pattern. This is NOT proof of attention.
    focus_parts = [
        c for c in ["EEI_rz", "negative_FFD_variability",
                    "negative_TRT_variability"] if c in df.columns
    ]
    df["focus_like_evidence"] = row_mean(df, focus_parts)

    # Fatigue-compatible EEG pattern: theta and alpha elevation with lower EEI.
    # Missing blink/PERCLOS/eyelid closure means this cannot establish fatigue.
    fatigue_parts = [
        c for c in ["ThetaPower_rz", "AlphaPower_rz", "low_EEI_evidence"]
        if c in df.columns
    ]
    df["fatigue_compatible_evidence"] = row_mean(df, fatigue_parts)

    # Attention instability / disengagement-like: low EEI + local variability.
    instability_parts = [
        c for c in ["low_EEI_evidence", "EEI_local_sd_rz",
                    "FFD_local_sd_rz", "TRT_local_sd_rz"] if c in df.columns
    ]
    df["attention_instability_evidence"] = row_mean(df, instability_parts)

    # Workload: theta plus pupil and reading-time indicators. FMT is not repeated.
    workload_parts = [
        c for c in ["ThetaPower_rz", "PupilSize_rz", "GD_rz", "TRT_rz"]
        if c in df.columns
    ]
    df["workload_evidence"] = row_mean(df, workload_parts)

    # State assignment is intentionally conservative and permits "mixed".
    score_cols = [
        "focus_like_evidence", "fatigue_compatible_evidence",
        "attention_instability_evidence", "workload_evidence"
    ]
    labels, strengths, top_scores, second_scores = [], [], [], []
    for _, row in df[score_cols].iterrows():
        valid = {k: float(row[k]) for k in score_cols if pd.notna(row[k])}
        ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            labels.append("Insufficient data")
            strengths.append("Unavailable")
            top_scores.append(np.nan)
            second_scores.append(np.nan)
            continue
        top_name, top_value = ranked[0]
        second_value = ranked[1][1] if len(ranked) > 1 else -np.inf
        top_scores.append(top_value)
        second_scores.append(second_value if np.isfinite(second_value) else np.nan)
        if top_value < EVIDENCE_THRESHOLD:
            label = "Mixed / indeterminate"
        elif top_value - second_value < AMBIGUITY_MARGIN:
            label = "Mixed / overlapping evidence"
        else:
            label = {
                "focus_like_evidence": "Engagement-supported / focused-like",
                "fatigue_compatible_evidence": "Fatigue-compatible EEG pattern",
                "attention_instability_evidence": "Attention-instability / disengagement-like",
                "workload_evidence": "High cognitive-workload-like pattern",
            }[top_name]
        labels.append(label)
        strengths.append(
            "Stronger heuristic evidence" if top_value >= STRONG_EVIDENCE_THRESHOLD
            else "Moderate heuristic evidence" if top_value >= EVIDENCE_THRESHOLD
            else "Weak / no clear evidence"
        )

    df["cognitive_pattern"] = labels
    df["pattern_evidence_strength"] = strengths
    df["top_pattern_score"] = top_scores
    df["second_pattern_score"] = second_scores
    return df


def interpret_cluster_scores(row):
    """Map cluster-average evidence scores to a cautious descriptive name."""
    score_map = {
        "focus_like_evidence": "Engagement-supported / focused-like pattern",
        "fatigue_compatible_evidence": "Fatigue-compatible EEG pattern",
        "attention_instability_evidence": "Attention-instability / disengagement-like pattern",
        "workload_evidence": "High cognitive-workload-like pattern",
    }
    available = [(k, row.get(k, np.nan)) for k in score_map]
    available = [(k, float(v)) for k, v in available if pd.notna(v)]
    if not available:
        return "Insufficient feature evidence", "No usable composite scores"
    if int(row.get("sentence_count", 0)) < 10:
        return "Too few sentences for interpretation", "Cluster has fewer than 10 sentences; profile is unstable"
    available.sort(key=lambda x: x[1], reverse=True)
    first = available[0]
    second = available[1][1] if len(available) > 1 else -np.inf
    if first[1] < EVIDENCE_THRESHOLD:
        return "Mixed / indeterminate pattern", "No evidence score reached heuristic threshold"
    if first[1] - second < AMBIGUITY_MARGIN:
        return "Mixed / overlapping pattern", "Top evidence scores are too close"
    return score_map[first[0]], f"Highest score: {first[0]}={first[1]:.2f} robust-z units"


def personal_reference_ranges(df):
    """Median +/- 1.5 robust SD; typical session range, not optimal range."""
    rows = []
    for pid, sub in df.groupby("patient_id", dropna=False, sort=False):
        for feature in FEATURES:
            if feature not in sub.columns:
                continue
            x = pd.to_numeric(sub[feature], errors="coerce").dropna()
            if x.empty:
                continue

            median = float(x.median())
            mad = float((x - median).abs().median())
            robust_sd = 1.4826 * mad

            if robust_sd < EPS:
                lower = upper = median
            else:
                lower = median - REFERENCE_MAD_MULTIPLIER * robust_sd
                upper = median + REFERENCE_MAD_MULTIPLIER * robust_sd

            rows.append({
                "patient_id": pid,
                "feature": feature,
                "session_median": median,
                "personal_reference_lower": lower,
                "personal_reference_upper": upper,
                "meaning": "Typical range in this session; not an optimal range"
            })

    return pd.DataFrame(rows)


def analyse_clusters(df):
    """
    Describe GMM cluster feature profiles relative to participant baseline.
    This does NOT assign definitive psychological state labels.
    """
    if "cluster_id" not in df.columns:
        return None, ["No cluster_id column found in input CSV."]

    features = [f for f in FEATURES if f in df.columns]
    if not features:
        return None, ["No usable feature columns for cluster analysis."]

    work = df.copy()
    for feature in features:
        work[feature + "_person_z"] = (
            work.groupby("patient_id", sort=False)[feature]
            .transform(robust_z)
        )

    group_cols = ["patient_id", "cluster_id"]
    stats_cols = features + [f + "_person_z" for f in features]

    score_cols = [c for c in [
        "focus_like_evidence", "fatigue_compatible_evidence",
        "attention_instability_evidence", "workload_evidence"
    ] if c in work.columns]
    stats = (
        work.groupby(group_cols, dropna=False)[stats_cols + score_cols]
        .mean()
        .reset_index()
    )
    counts = (
        work.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="sentence_count")
    )
    stats = stats.merge(counts, on=group_cols, how="left")
    if score_cols:
        interpretations = stats.apply(interpret_cluster_scores, axis=1, result_type="expand")
        interpretations.columns = ["heuristic_cluster_interpretation", "interpretation_reason"]
        stats = pd.concat([stats, interpretations], axis=1)
    stats["percentage"] = (
        100 * stats["sentence_count"]
        / stats.groupby("patient_id")["sentence_count"].transform("sum")
    )

    if "cluster_name" in work.columns:
        names = (
            work.groupby(group_cols)["cluster_name"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
            .reset_index()
        )
        stats = stats.merge(names, on=group_cols, how="left")

    lines = [
        "CLUSTER ASSIGNMENT AND FEATURE-BASED INTERPRETATION",
        "-" * 76,
        "Cluster membership tells which statistical pattern each sentence resembles.",
        "Cluster IDs are arbitrary; they do not inherently mean focus, fatigue or distraction.",
        ""
    ]

    for pid, sub in stats.groupby("patient_id", sort=False):
        dominant = sub.loc[sub["sentence_count"].idxmax()]
        lines.append(f"PARTICIPANT: {pid}")
        lines.append(
            f"DOMINANT CLUSTER: {dominant['cluster_id']} "
            f"({int(dominant['sentence_count'])} sentences; "
            f"{dominant['percentage']:.1f}% of session)"
        )
        if "cluster_name" in dominant.index:
            lines.append(f"Cluster name: {dominant['cluster_name']}")
        lines.append("")
        lines.append("Cluster-by-cluster profile:")

        for _, row in sub.sort_values(
            "sentence_count", ascending=False
        ).iterrows():
            cid = row["cluster_id"]
            cname = row.get("cluster_name", "")
            title = f"Cluster {cid}"
            if pd.notna(cname) and str(cname):
                title += f" ({cname})"

            lines.append(
                f"\n{title}: {int(row['sentence_count'])} sentences "
                f"({row['percentage']:.1f}%)"
            )

            deviations = []
            for feature in features:
                z = row.get(feature + "_person_z", np.nan)
                if pd.notna(z):
                    deviations.append((feature, float(z)))
            deviations.sort(key=lambda pair: abs(pair[1]), reverse=True)

            if "heuristic_cluster_interpretation" in row.index:
                lines.append(
                    "  HEURISTIC INTERPRETATION: "
                    + str(row["heuristic_cluster_interpretation"])
                )
                lines.append("  Reason: " + str(row.get("interpretation_reason", "")))
                lines.append(
                    "  Evidence scores (robust-z units; not probabilities): "
                    + ", ".join(
                        f"{c}={row[c]:.2f}" for c in [
                            "focus_like_evidence", "fatigue_compatible_evidence",
                            "attention_instability_evidence", "workload_evidence"
                        ] if c in row.index and pd.notna(row[c])
                    )
                )
            lines.append("  Strongest feature differences from participant median:")
            for feature, z in deviations[:5]:
                direction = "above" if z > 0 else "below"
                lines.append(
                    f"    {feature}: {abs(z):.2f} robust SD {direction} median"
                )

            # Carefully qualified pattern description.
            eei = row.get("EEI_person_z", np.nan)
            theta = row.get("ThetaPower_person_z", np.nan)
            pupil = row.get("PupilSize_person_z", np.nan)
            gd = row.get("GD_person_z", np.nan)
            trt = row.get("TRT_person_z", np.nan)

            lines.append("  Possible interpretation of feature pattern:")
            if pd.notna(eei) and eei > 0.5:
                lines.append(
                    "    Higher-than-usual EEG engagement-ratio feature. "
                    "This may be consistent with stronger engagement-related activity."
                )
            elif pd.notna(eei) and eei < -0.5:
                lines.append(
                    "    Lower-than-usual EEG engagement-ratio feature. "
                    "This may warrant review but does not prove distraction."
                )
            else:
                lines.append("    EEG engagement-ratio feature is near the session level.")

            if pd.notna(theta) and theta > 0.5:
                lines.append(
                    "    Higher theta-related feature; may reflect cognitive effort "
                    "or other state/task effects."
                )
            if (
                pd.notna(pupil) and pd.notna(gd) and pd.notna(trt)
                and pupil > 0.5 and gd > 0.5 and trt > 0.5
            ):
                lines.append(
                    "    Larger pupil and longer reading times co-occur; "
                    "could be consistent with greater processing demand."
                )
            elif (
                pd.notna(gd) and pd.notna(trt)
                and gd > 0.5 and trt > 0.5
            ):
                lines.append(
                    "    Longer gaze and total reading times; may reflect "
                    "rereading or greater processing demand."
                )

            lines.append(
                "    The heuristic label above is a feature-pattern description only. "
                "It is not a verified mental state; fatigue and mind-wandering evidence "
                "is especially limited by missing blink/PERCLOS/thought-probe measures."
            )

        lines.append("")

    return stats, lines


def build_session_conclusion(df, cluster_stats):
    lines = [
        "OVERALL SESSION CONCLUSION",
        "-" * 76,
        "This is a descriptive interpretation of the available signals, not a diagnosis.",
        ""
    ]

    for pid, sub in df.groupby("patient_id", sort=False):
        lines.append(f"Participant: {pid}")

        dominant_text = "Not available"
        if cluster_stats is not None:
            pstats = cluster_stats[cluster_stats["patient_id"] == pid]
            if not pstats.empty:
                dominant = pstats.loc[pstats["sentence_count"].idxmax()]
                dominant_text = (
                    f"Cluster {dominant['cluster_id']} "
                    f"({int(dominant['sentence_count'])} sentences, "
                    f"{dominant['percentage']:.1f}%)"
                )

        lines.append(f"Most frequent GMM pattern: {dominant_text}")

        pattern_counts = sub["cognitive_pattern"].value_counts(dropna=False)
        lines.append("Heuristic cognitive-pattern distribution:")
        for pattern, count in pattern_counts.items():
            lines.append(f"  {pattern}: {int(count)} sentences ({100*count/len(sub):.1f}%)")
        if not pattern_counts.empty:
            top_pattern = pattern_counts.index[0]
            lines.append(
                "Most frequent heuristic pattern: " + str(top_pattern)
                + " (descriptive evidence label, not a verified mental state)"
            )

        means = {}
        for raw, label, pct, category in INDEX_INFO:
            means[raw] = pd.to_numeric(
                sub[pct], errors="coerce"
            ).mean() if pct in sub.columns else np.nan

        lines.append(
            "Session-relative index averages: "
            + ", ".join(
                f"{key}={value:.1f}/100 percentile"
                for key, value in means.items()
            )
        )

        rei = means["REI"]
        sai = means["SAI"]
        mwli = means["MWLI_proxy"]
        cwi = means["CWI"]
        vsi = means["VSI_proxy"]

        if pd.notna(rei) and pd.notna(sai):
            if rei >= 66.7 and sai >= 66.7:
                lines.append(
                    "Interpretation: relatively high engagement-related evidence "
                    "and stability compared with this participant's own session."
                )
            elif rei <= 33.3 and sai <= 33.3:
                lines.append(
                    "Interpretation: relatively low engagement-related evidence "
                    "and stability compared with this participant's own session. "
                    "Check signal quality and task context."
                )
            else:
                lines.append(
                    "Interpretation: mixed engagement/stability profile; "
                    "cannot conclude focused or unfocused."
                )

        if pd.notna(mwli) and mwli >= 66.7:
            lines.append(
                "Mind-wandering-like proxy is relatively high; this is not "
                "confirmation of mind wandering."
            )
        if pd.notna(cwi) and cwi >= 66.7:
            lines.append(
                "Workload-related evidence is relatively high; high workload "
                "can occur during focused effort."
            )
        if pd.notna(vsi) and vsi <= 33.3:
            lines.append(
                "Signal-stability proxy is relatively low; this does not "
                "establish fatigue or reduced alertness."
            )

        lines.append("")

    lines.extend([
        "A true focus/fatigue classification requires independent labels such as",
        "thought probes, comprehension/task performance, and appropriate fatigue",
        "measures. Do not convert GMM cluster IDs directly into cognitive states."
    ])
    return lines


def write_report(df, reference_df, cluster_stats, cluster_lines, output_dir):
    lines = [
        "ZUCO COGNITIVE PROFILE REPORT",
        "=" * 76,
        f"Sentence rows: {len(df)}",
        f"Participants: {df['patient_id'].nunique()}",
        "",
        "WHAT EACH INDEX MEANS",
        "-" * 76,
        "1. SAI - Engagement-Stability Evidence Proxy:",
        "   Engagement-related EEG evidence minus local signal/reading variability.",
        "   It describes relative stability; it does not measure sustained attention directly.",
        "",
        "2. REI - Engagement-Related Signal Index:",
        "   Combines EEI, theta-related activity and pupil size as exploratory cues.",
        "   It is not a direct measure of reading engagement or concentration.",
        "",
        "3. MWLI_proxy - Attention-Instability Evidence Proxy:",
        "   Combines inverse EEI-related evidence and local reading/fixation variability.",
        "   It does not identify distraction or confirm mind wandering.",
        "",
        "4. CWI - Cognitive Workload Evidence Index:",
        "   Combines theta-related features, pupil size, gaze duration and total reading",
        "   time. High workload can occur during focused reading too.",
        "",
        "5. VSI_proxy - Signal-Stability Proxy:",
        "   Describes consistency of available EEG/eye features. It is not a validated",
        "   vigilance, fatigue or alertness measure.",
        "",
        "OVERALL SESSION VALUES",
        "-" * 76,
        f"{'Index':<42} {'Mean raw':>12} {'Mean pct':>12}",
        "-" * 76
    ]

    for raw, label, pct, category in INDEX_INFO:
        raw_mean = pd.to_numeric(df[raw], errors="coerce").mean()
        pct_mean = pd.to_numeric(df[pct], errors="coerce").mean()
        lines.append(f"{label:<42} {raw_mean:>12.3f} {pct_mean:>12.1f}")

    lines.extend([
        "",
        "Mean percentile is within-participant session rank, NOT a concentration percentage.",
        "",
        "RELATIVE CATEGORY COUNTS",
        "-" * 76
    ])

    for raw, label, pct, category in INDEX_INFO:
        lines.append(f"\n{label}:")
        for name, count in df[category].value_counts(dropna=False).items():
            share = 100 * count / len(df) if len(df) else 0
            lines.append(f"  {name}: {count} sentences ({share:.1f}%)")

    lines.extend([
        "",
        "HEURISTIC COGNITIVE-PATTERN INTERPRETATION",
        "-" * 76,
        "The following are cautious pattern labels derived from participant-relative",
        "robust-z feature scores. They are not supervised predictions or validated states.",
        f"Evidence threshold: {EVIDENCE_THRESHOLD:.2f} robust-z units; stronger evidence: {STRONG_EVIDENCE_THRESHOLD:.2f}.",
        f"Ambiguity margin: {AMBIGUITY_MARGIN:.2f}; close competing scores are labelled mixed.",
        "",
        "Score formulas (equal-weight mean of available components):",
        "  Focus-like = mean(EEI_z, -local_FFD_variability_z, -local_TRT_variability_z).",
        "  Fatigue-compatible = mean(Theta_z, Alpha_z, -EEI_z).",
        "  Attention-instability = mean(-EEI_z, local_EEI_variability_z,",
        "      local_FFD_variability_z, local_TRT_variability_z).",
        "  Workload-like = mean(Theta_z, Pupil_z, GD_z, TRT_z).",
        "FMT is not counted separately because in the supplied CSV it duplicates ThetaPower.",
        "These scores are heuristic and overlapping; they are not probabilities.",
        "",
        "SENTENCE-LEVEL COGNITIVE PATTERN COUNTS",
        "-" * 76,
        *[f"  {name}: {count} sentences ({100*count/len(df):.1f}%)"
          for name, count in df["cognitive_pattern"].value_counts(dropna=False).items()],
        "",
        "PERSONAL REFERENCE RANGES FOR INPUT FEATURES",
        "-" * 76,
        "These describe typical values in this participant's current session.",
        "They are NOT optimal, healthy or clinical ranges.",
        "",
        f"{'Participant':<22} {'Feature':<18} {'Median':>12} {'Lower':>12} {'Upper':>12}"
    ])

    for _, row in reference_df.iterrows():
        lines.append(
            f"{str(row['patient_id']):<22} {row['feature']:<18} "
            f"{row['session_median']:>12.4f} "
            f"{row['personal_reference_lower']:>12.4f} "
            f"{row['personal_reference_upper']:>12.4f}"
        )

    lines.extend([
        "",
        "OPTIMAL VALUES / TARGETS",
        "-" * 76,
        "No universal optimal values can be responsibly assigned from this CSV.",
        "EEG values depend on electrodes, reference, preprocessing and units.",
        "Eye measures depend on participant, lighting, text difficulty and task.",
        "Personal ranges are descriptive baselines, not targets or abnormality limits.",
        "",
        *cluster_lines,
        "",
        *build_session_conclusion(df, cluster_stats),
        "",
        "DATA AND MODEL LIMITATIONS",
        "-" * 76,
        "- FMT may be whole-scalp theta proxy, not channel-specific frontal-midline theta.",
        "- FixationCount may be mean word-level nFixations, not total sentence fixations.",
        "- CSV may lack regression counts, skipping, blink rate, PERCLOS and baseline-corrected pupil response.",
        "- GMM cluster IDs/names are arbitrary statistical groups.",
        "- cluster_probability is GMM membership strength, not probability of concentration.",
        "- Sentence index is sequence order, not clock time.",
        "- All proposed composite indices require validation against independent labels.",
        "- Focus-like evidence is based on EEI and short-window stability, not direct attention measurement.",
        "- Fatigue-compatible score uses theta/alpha/EEI only; blink rate, PERCLOS, eyelid closure,",
        "  and time-on-task trend are absent, so do not report a definitive fatigue classification.",
        "- Mind-wandering research reports heterogeneous eye-movement effects; GD/TRT/FFD alone",
        "  cannot identify distraction. Thought probes and skipping/regression/saccade features help.",
        "- Pupil size is sensitive to luminance and baseline; it is treated only as a workload cue.",
        "- Workload and fatigue overlap physiologically; high workload may occur during focused effort.",
        "- Thresholds 0.50 and 1.00 are transparent engineering heuristics in robust-z units,",
        "  not thresholds established by clinical or cognitive neuroscience studies.",
        "",
        "SELECTED RESEARCH BASIS",
        "- EEG fatigue meta-analysis: https://pubmed.ncbi.nlm.nih.gov/32108954/",
        "- Mental-fatigue physiological-index review: https://pmc.ncbi.nlm.nih.gov/articles/PMC11865143/",
        "- Mind-wandering eye-movement meta-analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC13132886/",
        "- EEG workload meta-analysis: https://pubmed.ncbi.nlm.nih.gov/35128686/",
        "- EEG + eye-tracking systematic review: https://pmc.ncbi.nlm.nih.gov/articles/PMC12895110/",
        "- ZuCo 2.0 dataset description: https://aclanthology.org/2020.lrec-1.18/"
    ])

    report = "\n".join(lines)
    (output_dir / "cognitive_profile_summary.txt").write_text(
        report, encoding="utf-8"
    )
    return report


def make_plots(df, cluster_stats, output_dir):
    participants = list(df["patient_id"].dropna().unique())

    # Plot 1: index timeline
    plot_specs = [
        ("SAI_percentile", "SAI - attention stability proxy"),
        ("REI_percentile", "REI - engagement evidence"),
        ("MWLI_percentile", "MWLI - mind-wandering-like proxy"),
        ("CWI_percentile", "CWI - workload evidence"),
        ("VSI_percentile", "VSI - signal stability")
    ]

    fig, axes = plt.subplots(
        len(participants), 1,
        figsize=(14, max(4, 3.2 * len(participants))),
        squeeze=False
    )

    for ax, pid in zip(axes[:, 0], participants):
        sub = df[df["patient_id"] == pid]
        x = pd.to_numeric(
            sub["sentence_index"], errors="coerce"
        ).to_numpy(dtype=float)

        for col, label in plot_specs:
            y = pd.to_numeric(
                sub[col], errors="coerce"
            ).to_numpy(dtype=float)
            if np.isfinite(y).any():
                ax.plot(x, y, marker=".", linewidth=1.1,
                        markersize=3, label=label)

        ax.set_title(f"Participant: {pid}")
        ax.set_xlabel("Sentence index (sequence order, not clock time)")
        ax.set_ylabel("Within-session percentile (0-100)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(
        output_dir / "cognitive_profile_timeline.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 2: category timeline
    cat_specs = [
        ("SAI_category", "SAI"),
        ("REI_category", "REI"),
        ("MWLI_category", "MWLI proxy"),
        ("CWI_category", "CWI"),
        ("VSI_category", "VSI")
    ]

    fig, axes = plt.subplots(
        len(participants), 1,
        figsize=(14, max(4, 2.5 * len(participants))),
        squeeze=False
    )

    for ax, pid in zip(axes[:, 0], participants):
        sub = df[df["patient_id"] == pid]
        x = pd.to_numeric(
            sub["sentence_index"], errors="coerce"
        ).to_numpy(dtype=float)

        for row, (col, label) in enumerate(cat_specs):
            cats = sub[col].astype(str)
            vals = np.full(len(cats), np.nan)
            vals[cats.str.startswith(("Lower", "Less")).to_numpy()] = 0
            vals[cats.str.startswith("Moderate").to_numpy()] = 1
            vals[cats.str.startswith(("Higher", "More")).to_numpy()] = 2

            ax.scatter(
                x, np.full(len(x), row), c=vals,
                cmap="viridis", vmin=0, vmax=2,
                marker="s", s=25
            )

        ax.set_yticks(range(len(cat_specs)))
        ax.set_yticklabels([label for _, label in cat_specs])
        ax.set_xlabel("Sentence index (sequence order, not clock time)")
        ax.set_title(f"Relative category timeline: {pid}")
        ax.set_ylim(len(cat_specs) - 0.5, -0.5)
        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        output_dir / "cognitive_state_timeline.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot 3: sentence-level heuristic cognitive pattern distribution
    if "cognitive_pattern" in df.columns:
        counts = df["cognitive_pattern"].value_counts()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title("Heuristic Cognitive-Pattern Distribution (not validated states)")
        ax.set_ylabel("Sentence count")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "cognitive_pattern_distribution.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Plot 4: cluster distribution
    cluster_col = (
        "cluster_name" if "cluster_name" in df.columns
        else "cluster_id" if "cluster_id" in df.columns
        else None
    )

    if cluster_col:
        counts = (
            df[cluster_col].fillna("Missing").astype(str).value_counts()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index, counts.values)
        ax.set_title("GMM Cluster Distribution (statistical groups)")
        ax.set_xlabel(cluster_col)
        ax.set_ylabel("Sentence count")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "cluster_distribution.png",
            dpi=180, bbox_inches="tight"
        )
        plt.close(fig)


def main():
    input_path = Path(INPUT_CSV)
    output_dir = Path(OUTPUT_DIR)

    if not input_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {input_path}\n"
            "Edit INPUT_CSV at the top of this script."
        )

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    df = prepare_data(df)
    df = add_indices(df)
    df = add_cognitive_evidence_scores(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sentence-level results.
    df.to_csv(
        output_dir / "cognitive_profile_output.csv",
        index=False
    )
    df.to_csv(
        output_dir / "cognitive_state_predictions.csv",
        index=False
    )

    # Save personal reference ranges.
    reference_df = personal_reference_ranges(df)
    reference_df.to_csv(
        output_dir / "personal_reference_ranges.csv",
        index=False
    )

    # Cluster profile analysis and report.
    cluster_stats, cluster_lines = analyse_clusters(df)
    if cluster_stats is not None:
        cluster_stats.to_csv(
            output_dir / "cluster_feature_analysis.csv",
            index=False
        )
        keep = [c for c in [
            "patient_id", "cluster_id", "cluster_name", "sentence_count", "percentage",
            "heuristic_cluster_interpretation", "interpretation_reason",
            "focus_like_evidence", "fatigue_compatible_evidence",
            "attention_instability_evidence", "workload_evidence"
        ] if c in cluster_stats.columns]
        cluster_stats[keep].to_csv(
            output_dir / "cluster_cognitive_interpretation.csv", index=False
        )

    # Create plots.
    make_plots(df, cluster_stats, output_dir)

    # Write and print complete text report.
    report = write_report(
        df, reference_df, cluster_stats, cluster_lines, output_dir
    )
    print("\n" + report)

    print("\nFILES GENERATED")
    print("-" * 60)
    for filename in [
        "cognitive_profile_output.csv",
        "cognitive_profile_summary.txt",
        "personal_reference_ranges.csv",
        "cluster_feature_analysis.csv",
        "cognitive_profile_timeline.png",
        "cognitive_state_timeline.png",
        "cluster_distribution.png",
        "cognitive_state_predictions.csv",
        "cluster_cognitive_interpretation.csv",
        "cognitive_pattern_distribution.png"
    ]:
        file_path = output_dir / filename
        if file_path.exists():
            print(file_path)

    print("\n[DONE] Cognitive profile generated.")
    print("Reminder: these are exploratory patterns, not validated diagnoses.")


if __name__ == "__main__":
    main()
