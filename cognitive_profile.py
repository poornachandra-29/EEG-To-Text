#!/usr/bin/env python3

"""
ZUCO COGNITIVE PROFILE ANALYSIS

Input: Sentence-level ZuCo CSV.

Outputs:
    cognitive_profile_output.csv
    cognitive_profile_summary.txt
    personal_reference_ranges.csv
    cluster_feature_analysis.csv
    cognitive_profile_timeline.png
    cognitive_state_timeline.png
    cluster_distribution.png

Install:
    pip install pandas numpy matplotlib

Run:
    python3 cognitive_profile.py

IMPORTANT:
These are exploratory indices, not validated clinical measures.
GMM clusters are statistical patterns, not automatic cognitive-state labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. USER SETTINGS
# ============================================================

INPUT_CSV = r"/mnt/DATA/SPLAB/1. Code/EEG-To-Text/test_nrei_csv/test_predictions.csv"  # CHANGE THIS PATH

OUTPUT_DIR = r"cognitive_profile_results"

ROLLING_WINDOW = 10
MIN_PERIODS = 3
REFERENCE_MAD_MULTIPLIER = 1.5

EPS = 1e-9


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "EEI", "FMT", "ASM", "GD", "TRT", "FFD",
    "PupilSize", "FixationCount",
    "ThetaPower", "AlphaPower", "BetaPower"
]

INDEX_INFO = [
    ("SAI", "Sustained Attention Stability Index",
     "SAI_percentile", "SAI_category"),
    ("REI", "Reading Engagement Evidence Index",
     "REI_percentile", "REI_category"),
    ("MWLI_proxy", "Mind-Wandering-Like Evidence Proxy",
     "MWLI_percentile", "MWLI_category"),
    ("CWI", "Cognitive Workload Evidence Index",
     "CWI_percentile", "CWI_category"),
    ("VSI_proxy", "Vigilance Signal-Stability Proxy",
     "VSI_percentile", "VSI_category"),
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def robust_z(series):
    """Robust Z-score using median and MAD."""

    x = pd.to_numeric(
        series, errors="coerce"
    ).astype(float)

    median = x.median()
    mad = (x - median).abs().median()

    if pd.isna(median):
        return pd.Series(np.nan, index=x.index, dtype=float)

    if pd.isna(mad) or mad < EPS:
        std = x.std()

        if pd.isna(std) or std < EPS:
            return pd.Series(
                np.where(x.notna(), 0.0, np.nan),
                index=x.index,
                dtype=float
            )

        return (x - x.mean()) / std

    return 0.67448975 * (x - median) / mad


def row_mean(df, columns):
    """Mean of available features for each row."""

    cols = [c for c in columns if c in df.columns]

    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)

    return df[cols].mean(axis=1, skipna=True)


def prepare_data(df):

    if "patient_id" not in df.columns:
        df["patient_id"] = "participant_1"

    if "sentence_index" not in df.columns:
        df["sentence_index"] = (
            df.groupby("patient_id").cumcount()
        )

    df["sentence_index"] = pd.to_numeric(
        df["sentence_index"], errors="coerce"
    )

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.sort_values(
        ["patient_id", "sentence_index"],
        kind="stable"
    ).reset_index(drop=True)

    if not any(c in df.columns for c in FEATURES):
        raise ValueError(
            "No expected ZuCo feature columns found."
        )

    return df


# ============================================================
# 4. FEATURE NORMALISATION
# ============================================================

def add_z_features(df):

    zcols = FEATURES.copy()

    if {"ThetaPower", "AlphaPower"}.issubset(df.columns):

        df["Theta_Alpha_Ratio"] = (
            df["ThetaPower"] /
            (df["AlphaPower"].abs() + EPS)
        )

        zcols.append("Theta_Alpha_Ratio")

    if {"ThetaPower", "BetaPower"}.issubset(df.columns):

        df["Theta_Beta_Ratio"] = (
            df["ThetaPower"] /
            (df["BetaPower"].abs() + EPS)
        )

        zcols.append("Theta_Beta_Ratio")

    for col in zcols:

        if col in df.columns:

            df[col + "_rz"] = (
                df.groupby("patient_id", sort=False)[col]
                .transform(robust_z)
            )

    return df


# ============================================================
# 5. ROLLING VARIABILITY
# ============================================================

def add_variability(df):

    pairs = [
        ("EEI", "EEI_local_sd"),
        ("FFD", "FFD_local_sd"),
        ("TRT", "TRT_local_sd"),
        ("PupilSize", "Pupil_local_sd")
    ]

    for source, target in pairs:

        if source in df.columns:

            df[target] = (
                df.groupby("patient_id", sort=False)[source]
                .transform(
                    lambda s: s.rolling(
                        ROLLING_WINDOW,
                        min_periods=MIN_PERIODS
                    ).std()
                )
            )

            df[target + "_rz"] = (
                df.groupby("patient_id", sort=False)[target]
                .transform(robust_z)
            )

    return df


# ============================================================
# 6. CALCULATE FIVE COGNITIVE INDICES
# ============================================================

def add_indices(df):

    df = add_z_features(df)

    # --------------------------------------------------------
    # REI: READING ENGAGEMENT EVIDENCE INDEX
    # --------------------------------------------------------

    # EEG engagement ratio + theta proxy + pupil response.
    # Higher value = stronger engagement-related evidence,
    # NOT confirmed concentration.

    df["REI"] = row_mean(
        df,
        ["EEI_rz", "FMT_rz", "PupilSize_rz"]
    )

    # --------------------------------------------------------
    # CWI: COGNITIVE WORKLOAD EVIDENCE INDEX
    # --------------------------------------------------------

    # Theta + pupil + gaze duration + total reading time.
    # Higher value = greater workload-related evidence.

    df["CWI"] = row_mean(
        df,
        [
            "FMT_rz",
            "ThetaPower_rz",
            "PupilSize_rz",
            "GD_rz",
            "TRT_rz"
        ]
    )

    df = add_variability(df)

    variability_cols = [
        c for c in [
            "EEI_local_sd_rz",
            "FFD_local_sd_rz",
            "TRT_local_sd_rz",
            "Pupil_local_sd_rz"
        ]
        if c in df.columns
    ]

    instability = row_mean(
        df, variability_cols
    )

    # --------------------------------------------------------
    # SAI: SUSTAINED ATTENTION STABILITY INDEX
    # --------------------------------------------------------

    # Engagement-related evidence minus local variability.

    df["SAI"] = df["REI"] - instability

    # --------------------------------------------------------
    # MWLI: MIND-WANDERING-LIKE EVIDENCE PROXY
    # --------------------------------------------------------

    reading_variability = row_mean(
        df,
        [
            "FFD_local_sd_rz",
            "TRT_local_sd_rz"
        ]
    )

    if "EEI_rz" in df.columns:

        df["MWLI_proxy"] = (
            -df["EEI_rz"] + reading_variability
        )

    else:

        df["MWLI_proxy"] = np.nan

    # --------------------------------------------------------
    # VSI: VIGILANCE SIGNAL-STABILITY PROXY
    # --------------------------------------------------------

    df["VSI_proxy"] = -instability

    # --------------------------------------------------------
    # WITHIN-PARTICIPANT PERCENTILE VALUES
    # --------------------------------------------------------

    pairs = [
        ("SAI", "SAI_percentile"),
        ("REI", "REI_percentile"),
        ("MWLI_proxy", "MWLI_percentile"),
        ("CWI", "CWI_percentile"),
        ("VSI_proxy", "VSI_percentile")
    ]

    for raw, percentile in pairs:

        df[percentile] = (
            df.groupby("patient_id", sort=False)[raw]
            .rank(pct=True, method="average") * 100
        )

    # --------------------------------------------------------
    # RELATIVE CATEGORIES
    # --------------------------------------------------------

    specifications = [
        (
            "SAI", "SAI_category",
            "Lower relative stability",
            "Moderate relative stability",
            "Higher relative stability"
        ),
        (
            "REI", "REI_category",
            "Lower engagement-related evidence",
            "Moderate engagement-related evidence",
            "Higher engagement-related evidence"
        ),
        (
            "MWLI_proxy", "MWLI_category",
            "Lower drift-like evidence",
            "Moderate drift-like evidence",
            "Higher drift-like evidence"
        ),
        (
            "CWI", "CWI_category",
            "Lower workload-related evidence",
            "Moderate workload-related evidence",
            "Higher workload-related evidence"
        ),
        (
            "VSI_proxy", "VSI_category",
            "Lower signal stability",
            "Moderate signal stability",
            "Higher signal stability"
        )
    ]

    for source, target, low, middle, high in specifications:

        def make_labels(s):

            valid = s.dropna()

            if len(valid) < 3:
                return pd.Series(
                    "Insufficient data", index=s.index
                )

            q1 = valid.quantile(1 / 3)
            q2 = valid.quantile(2 / 3)

            labels = np.select(
                [s <= q1, s >= q2],
                [low, high],
                default=middle
            )

            return pd.Series(
                labels, index=s.index
            ).where(
                s.notna(), "Insufficient data"
            )

        df[target] = (
            df.groupby("patient_id", sort=False)[source]
            .transform(make_labels)
        )

    return df


# ============================================================
# 7. PERSONAL REFERENCE RANGES
# ============================================================

def personal_reference_ranges(df):

    rows = []

    for pid, sub in df.groupby(
        "patient_id", dropna=False, sort=False
    ):

        for feature in FEATURES:

            if feature not in sub.columns:
                continue

            x = pd.to_numeric(
                sub[feature], errors="coerce"
            ).dropna()

            if x.empty:
                continue

            median = float(x.median())

            mad = float(
                (x - median).abs().median()
            )

            robust_sd = 1.4826 * mad

            if robust_sd < EPS:
                lower = upper = median
            else:
                lower = (
                    median -
                    REFERENCE_MAD_MULTIPLIER * robust_sd
                )
                upper = (
                    median +
                    REFERENCE_MAD_MULTIPLIER * robust_sd
                )

            rows.append({
                "patient_id": pid,
                "feature": feature,
                "session_median": median,
                "personal_reference_lower": lower,
                "personal_reference_upper": upper,
                "meaning": "Typical session range, NOT optimal range"
            })

    return pd.DataFrame(rows)


# ============================================================
# 8. GMM CLUSTER FEATURE ANALYSIS
# ============================================================

def analyse_clusters(df):
    """
    Analyse cluster feature profiles.

    The function identifies the dominant cluster and describes
    its feature characteristics relative to the participant's
    own session.

    It does NOT invent focus/fatigue labels.
    """

    if "cluster_id" not in df.columns:

        return None, [
            "No cluster_id column found in CSV."
        ]

    features = [
        f for f in FEATURES
        if f in df.columns
    ]

    if not features:

        return None, [
            "No usable feature columns for cluster analysis."
        ]

    work = df.copy()

    # Participant-relative Z scores.

    for feature in features:

        work[feature + "_person_z"] = (
            work.groupby("patient_id", sort=False)[feature]
            .transform(robust_z)
        )

    group_cols = [
        "patient_id",
        "cluster_id"
    ]

    stats_cols = (
        features +
        [f + "_person_z" for f in features]
    )

    stats = (
        work.groupby(
            group_cols,
            dropna=False
        )[stats_cols]
        .mean()
        .reset_index()
    )

    counts = (
        work.groupby(
            group_cols,
            dropna=False
        )
        .size()
        .reset_index(name="sentence_count")
    )

    stats = stats.merge(
        counts,
        on=group_cols,
        how="left"
    )

    stats["percentage"] = (
        100 * stats["sentence_count"] /
        stats.groupby("patient_id")[
            "sentence_count"
        ].transform("sum")
    )

    if "cluster_name" in work.columns:

        names = (
            work.groupby(group_cols)["cluster_name"]
            .agg(
                lambda s:
                s.mode().iloc[0]
                if not s.mode().empty
                else "Unknown"
            )
            .reset_index()
        )

        stats = stats.merge(
            names,
            on=group_cols,
            how="left"
        )

    lines = [

        "CLUSTER ASSIGNMENT AND FEATURE-BASED INTERPRETATION",

        "-" * 76,

        "Cluster membership identifies statistical EEG/eye feature patterns.",

        "Cluster IDs do NOT automatically mean focus, fatigue or distraction.",

        ""

    ]

    for pid, sub in stats.groupby(
        "patient_id", sort=False
    ):

        dominant = sub.loc[
            sub["sentence_count"].idxmax()
        ]

        lines.append(
            f"PARTICIPANT: {pid}"
        )

        lines.append(
            f"DOMINANT CLUSTER: {dominant['cluster_id']} "
            f"({int(dominant['sentence_count'])} sentences; "
            f"{dominant['percentage']:.1f}% of session)"
        )

        if "cluster_name" in dominant.index:

            lines.append(
                f"Cluster name: {dominant['cluster_name']}"
            )

        lines.append("")
        lines.append("CLUSTER-BY-CLUSTER FEATURE PROFILE:")

        for _, row in sub.sort_values(
            "sentence_count",
            ascending=False
        ).iterrows():

            cid = row["cluster_id"]

            cname = row.get("cluster_name", "")

            title = f"Cluster {cid}"

            if pd.notna(cname) and str(cname):
                title += f" ({cname})"

            lines.append(
                f"\n{title}: "
                f"{int(row['sentence_count'])} sentences "
                f"({row['percentage']:.1f}%)"
            )

            deviations = []

            for feature in features:

                z = row.get(
                    feature + "_person_z",
                    np.nan
                )

                if pd.notna(z):

                    deviations.append(
                        (feature, float(z))
                    )

            deviations.sort(
                key=lambda item: abs(item[1]),
                reverse=True
            )

            lines.append(
                "  Strongest feature differences from participant median:"
            )

            for feature, z in deviations[:5]:

                direction = (
                    "above" if z > 0 else "below"
                )

                lines.append(
                    f"    {feature}: {abs(z):.2f} robust SD "
                    f"{direction} median"
                )

            # Feature-based interpretation.

            eei = row.get("EEI_person_z", np.nan)

            theta = row.get(
                "ThetaPower_person_z", np.nan
            )

            pupil = row.get(
                "PupilSize_person_z", np.nan
            )

            gd = row.get("GD_person_z", np.nan)

            trt = row.get("TRT_person_z", np.nan)

            lines.append(
                "  Possible physiological pattern:"
            )

            if pd.notna(eei) and eei > 0.5:

                lines.append(
                    "    Higher EEG engagement-ratio feature. "
                    "May be consistent with stronger engagement-related activity."
                )

            elif pd.notna(eei) and eei < -0.5:

                lines.append(
                    "    Lower EEG engagement-ratio feature. "
                    "This does not prove distraction."
                )

            else:

                lines.append(
                    "    EEG engagement-ratio feature is near the "
                    "participant's session level."
                )

            if pd.notna(theta) and theta > 0.5:

                lines.append(
                    "    Higher theta-related feature; may reflect "
                    "cognitive effort or other task/state effects."
                )

            if (
                pd.notna(pupil)
                and pd.notna(gd)
                and pd.notna(trt)
                and pupil > 0.5
                and gd > 0.5
                and trt > 0.5
            ):

                lines.append(
                    "    Larger pupil and longer reading times co-occur. "
                    "Could be consistent with greater processing demand."
                )

            elif (
                pd.notna(gd)
                and pd.notna(trt)
                and gd > 0.5
                and trt > 0.5
            ):

                lines.append(
                    "    Longer gaze and total reading times; may reflect "
                    "rereading or greater processing demand."
                )

            lines.append(
                "    This pattern cannot be labelled focused, distracted "
                "or fatigued without independent validation."
            )

        lines.append("")

    return stats, lines


# ============================================================
# 9. OVERALL SESSION CONCLUSION
# ============================================================

def build_session_conclusion(df, cluster_stats):

    lines = [

        "OVERALL SESSION CONCLUSION",

        "-" * 76,

        "This is a descriptive interpretation, not a diagnosis.",

        ""

    ]

    for pid, sub in df.groupby(
        "patient_id", sort=False
    ):

        lines.append(
            f"Participant: {pid}"
        )

        dominant_text = "Not available"

        if cluster_stats is not None:

            pstats = cluster_stats[
                cluster_stats["patient_id"] == pid
            ]

            if not pstats.empty:

                dominant = pstats.loc[
                    pstats["sentence_count"].idxmax()
                ]

                dominant_text = (

                    f"Cluster {dominant['cluster_id']} "

                    f"({int(dominant['sentence_count'])} sentences, "

                    f"{dominant['percentage']:.1f}%)"

                )

        lines.append(
            f"Most frequent GMM pattern: {dominant_text}"
        )

        means = {}

        for raw, label, pct, category in INDEX_INFO:

            means[raw] = pd.to_numeric(
                sub[pct],
                errors="coerce"
            ).mean()

        lines.append(
            "Session-relative index averages: "
            +
            ", ".join(
                f"{key}={value:.1f} percentile"
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
                    "Interpretation: relatively high engagement-related "
                    "evidence and stability compared with this session."
                )

            elif rei <= 33.3 and sai <= 33.3:

                lines.append(
                    "Interpretation: relatively low engagement-related "
                    "evidence and stability. Check signal quality and context."
                )

            else:

                lines.append(
                    "Interpretation: mixed engagement/stability profile. "
                    "Cannot conclude focused or unfocused."
                )

        if pd.notna(mwli) and mwli >= 66.7:

            lines.append(
                "Mind-wandering-like proxy is relatively high; "
                "this does NOT confirm mind wandering."
            )

        if pd.notna(cwi) and cwi >= 66.7:

            lines.append(
                "Workload-related evidence is relatively high. "
                "High workload can occur during focused effort."
            )

        if pd.notna(vsi) and vsi <= 33.3:

            lines.append(
                "Signal-stability proxy is relatively low. "
                "This does NOT establish fatigue."
            )

        lines.append("")

    lines.extend([

        "A true focus/fatigue classification requires independent labels",

        "such as thought probes, comprehension/task performance and",

        "appropriate fatigue measures. Do not convert cluster IDs directly",

        "into cognitive-state labels."

    ])

    return lines


# ============================================================
# 10. TEXT REPORT
# ============================================================

def write_report(
    df,
    reference_df,
    cluster_stats,
    cluster_lines,
    output_dir
):

    lines = [

        "ZUCO COGNITIVE PROFILE REPORT",

        "=" * 76,

        f"Sentence rows: {len(df)}",

        f"Participants: {df['patient_id'].nunique()}",

        "",

        "WHAT EACH INDEX MEANS",

        "-" * 76,

        "1. SAI - Sustained Attention Stability Index:",

        "   Engagement-related evidence minus local variability.",

        "   Attempts to describe consistency across sentence sequence.",

        "   It does NOT prove sustained attention.",

        "",

        "2. REI - Reading Engagement Evidence Index:",

        "   Combines EEG engagement ratio, FMT/theta proxy and pupil size.",

        "   Describes engagement-related physiology, not direct concentration.",

        "",

        "3. MWLI - Mind-Wandering-Like Evidence Proxy:",

        "   Combines inverse EEI-related evidence and local reading/fixation",

        "   variability. Flags patterns to investigate; does not confirm wandering.",

        "",

        "4. CWI - Cognitive Workload Evidence Index:",

        "   Combines theta-related features, pupil size, gaze duration and",

        "   total reading time. High workload can occur during focused reading.",

        "",

        "5. VSI - Vigilance Signal-Stability Proxy:",

        "   Describes consistency of available EEG/eye features.",

        "   It is NOT a validated fatigue or alertness measure.",

        "",

        "OVERALL SESSION VALUES",

        "-" * 76,

        f"{'Index':<42} {'Mean raw':>12} {'Mean pct':>12}",

        "-" * 76

    ]

    for raw, label, pct, category in INDEX_INFO:

        raw_mean = pd.to_numeric(
            df[raw], errors="coerce"
        ).mean()

        pct_mean = pd.to_numeric(
            df[pct], errors="coerce"
        ).mean()

        lines.append(

            f"{label:<42} "
            f"{raw_mean:>12.3f} "
            f"{pct_mean:>12.1f}"

        )

    lines.extend([

        "",

        "Mean percentile is within-participant session rank,",

        "NOT a concentration percentage.",

        "",

        "RELATIVE CATEGORY COUNTS",

        "-" * 76

    ])

    for raw, label, pct, category in INDEX_INFO:

        lines.append(f"\n{label}:")

        counts = df[category].value_counts(
            dropna=False
        )

        for name, count in counts.items():

            share = (
                100 * count / len(df)
                if len(df) else 0
            )

            lines.append(
                f"  {name}: {count} sentences ({share:.1f}%)"
            )

    lines.extend([

        "",

        "PERSONAL REFERENCE RANGES FOR INPUT FEATURES",

        "-" * 76,

        "These describe typical values in this participant's session.",

        "They are NOT optimal, healthy or clinical ranges.",

        "",

        f"{'Participant':<22} {'Feature':<18} "
        f"{'Median':>12} {'Lower':>12} {'Upper':>12}"

    ])

    for _, row in reference_df.iterrows():

        lines.append(

            f"{str(row['patient_id']):<22} "

            f"{row['feature']:<18} "

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

        "- FMT may be whole-scalp theta proxy, not channel-specific frontal theta.",

        "- FixationCount may be mean word-level nFixations, not total sentence fixations.",

        "- CSV may lack regressions, skipping, blink rate, PERCLOS and baseline-corrected pupil response.",

        "- GMM cluster IDs/names are arbitrary statistical groups.",

        "- cluster_probability is GMM membership strength, not concentration probability.",

        "- Sentence index is sequence order, not clock time.",

        "- All proposed composite indices require validation against independent labels."

    ])

    report = "\n".join(lines)

    (output_dir / "cognitive_profile_summary.txt").write_text(
        report,
        encoding="utf-8"
    )

    return report


# ============================================================
# 11. VISUALISATIONS
# ============================================================

def make_plots(df, output_dir):

    participants = list(
        df["patient_id"].dropna().unique()
    )

    # --------------------------------------------------------
    # PLOT 1: INDEX TIMELINE
    # --------------------------------------------------------

    plot_specs = [

        ("SAI_percentile", "SAI - attention stability"),

        ("REI_percentile", "REI - engagement evidence"),

        ("MWLI_percentile", "MWLI - drift-like evidence"),

        ("CWI_percentile", "CWI - workload evidence"),

        ("VSI_percentile", "VSI - signal stability")

    ]

    fig, axes = plt.subplots(

        len(participants),

        1,

        figsize=(14, max(4, 3.2 * len(participants))),

        squeeze=False

    )

    for ax, pid in zip(
        axes[:, 0],
        participants
    ):

        sub = df[
            df["patient_id"] == pid
        ]

        x = pd.to_numeric(
            sub["sentence_index"],
            errors="coerce"
        ).to_numpy(dtype=float)

        for col, label in plot_specs:

            y = pd.to_numeric(
                sub[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            if np.isfinite(y).any():

                ax.plot(
                    x,
                    y,
                    marker=".",
                    linewidth=1.1,
                    markersize=3,
                    label=label
                )

        ax.set_title(
            f"Participant: {pid}"
        )

        ax.set_xlabel(
            "Sentence index (sequence order, not clock time)"
        )

        ax.set_ylabel(
            "Within-session percentile (0-100)"
        )

        ax.set_ylim(0, 100)

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend(
            fontsize=8,
            loc="best"
        )

    fig.tight_layout()

    fig.savefig(
        output_dir / "cognitive_profile_timeline.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PLOT 2: CATEGORY TIMELINE
    # --------------------------------------------------------

    category_specs = [

        ("SAI_category", "SAI"),

        ("REI_category", "REI"),

        ("MWLI_category", "MWLI proxy"),

        ("CWI_category", "CWI"),

        ("VSI_category", "VSI")

    ]

    fig, axes = plt.subplots(

        len(participants),

        1,

        figsize=(14, max(4, 2.5 * len(participants))),

        squeeze=False

    )

    for ax, pid in zip(
        axes[:, 0],
        participants
    ):

        sub = df[
            df["patient_id"] == pid
        ]

        x = pd.to_numeric(
            sub["sentence_index"],
            errors="coerce"
        ).to_numpy(dtype=float)

        for row, (col, label) in enumerate(category_specs):

            cats = sub[col].astype(str)

            values = np.full(
                len(cats),
                np.nan
            )

            values[
                cats.str.startswith(
                    ("Lower", "Less")
                ).to_numpy()
            ] = 0

            values[
                cats.str.startswith(
                    "Moderate"
                ).to_numpy()
            ] = 1

            values[
                cats.str.startswith(
                    ("Higher", "More")
                ).to_numpy()
            ] = 2

            ax.scatter(

                x,

                np.full(len(x), row),

                c=values,

                cmap="viridis",

                vmin=0,

                vmax=2,

                marker="s",

                s=25

            )

        ax.set_yticks(
            range(len(category_specs))
        )

        ax.set_yticklabels(
            [label for _, label in category_specs]
        )

        ax.set_xlabel(
            "Sentence index (sequence order, not clock time)"
        )

        ax.set_title(
            f"Relative category timeline: {pid}"
        )

        ax.set_ylim(
            len(category_specs) - 0.5,
            -0.5
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

    fig.tight_layout()

    fig.savefig(
        output_dir / "cognitive_state_timeline.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PLOT 3: GMM CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    cluster_col = (

        "cluster_name"
        if "cluster_name" in df.columns

        else "cluster_id"
        if "cluster_id" in df.columns

        else None

    )

    if cluster_col:

        counts = (
            df[cluster_col]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_title(
            "GMM Cluster Distribution (Statistical Groups)"
        )

        ax.set_xlabel(
            cluster_col
        )

        ax.set_ylabel(
            "Sentence count"
        )

        ax.tick_params(
            axis="x",
            rotation=35
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / "cluster_distribution.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# 12. MAIN
# ============================================================

def main():

    input_path = Path(
        INPUT_CSV
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"CSV not found: {input_path}\n"
            "Change INPUT_CSV at the top of this file."
        )

    df = pd.read_csv(
        input_path
    )

    if df.empty:

        raise ValueError(
            "The input CSV contains no rows."
        )

    print(
        "\n[INFO] Loaded CSV:",
        input_path
    )

    print(
        "[INFO] Sentence rows:",
        len(df)
    )

    df = prepare_data(df)

    # Calculate five indices.

    df = add_indices(df)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save sentence-level results.

    df.to_csv(
        output_dir / "cognitive_profile_output.csv",
        index=False
    )

    # Calculate personal reference ranges.

    reference_df = personal_reference_ranges(
        df
    )

    reference_df.to_csv(
        output_dir / "personal_reference_ranges.csv",
        index=False
    )

    # Analyse GMM clusters.

    cluster_stats, cluster_lines = analyse_clusters(
        df
    )

    if cluster_stats is not None:

        cluster_stats.to_csv(
            output_dir / "cluster_feature_analysis.csv",
            index=False
        )

    # Generate plots.

    print(
        "[INFO] Generating visualisations..."
    )

    make_plots(
        df,
        output_dir
    )

    # Generate report.

    report = write_report(
        df,
        reference_df,
        cluster_stats,
        cluster_lines,
        output_dir
    )

    # Print complete report in terminal.

    print("\n" + report)

    # List output files.

    print("\nFILES GENERATED")

    print("-" * 60)

    filenames = [

        "cognitive_profile_output.csv",

        "cognitive_profile_summary.txt",

        "personal_reference_ranges.csv",

        "cluster_feature_analysis.csv",

        "cognitive_profile_timeline.png",

        "cognitive_state_timeline.png",

        "cluster_distribution.png"

    ]

    for filename in filenames:

        file_path = output_dir / filename

        if file_path.exists():

            print(file_path)

    print("\n[DONE] Cognitive profile generated.")

    print(
        "Reminder: these are exploratory patterns, not validated diagnoses."
    )


if __name__ == "__main__":

    main()