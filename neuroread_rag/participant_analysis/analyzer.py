"""
Deterministic Participant Analysis Module.

Calculates comprehensive descriptive, robust, and sentence-level statistics
directly from the participant CSV. Never invents or estimates numbers via LLM.
Preserves strict scientific separation between physiological measurements and psychological states.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from neuroread_rag.config import (
    METRIC_DEFINITIONS,
    MANDATORY_DISCLAIMERS,
    DEFAULT_PROFILE_CSV,
    FALLBACK_PREDICTIONS_CSV,
    PERSONAL_RANGES_CSV,
    CLUSTER_INTERPRETATION_CSV,
    OUTPUT_DIR,
    get_active_participant_csv,
    set_active_participant_csv
)
from neuroread_rag.participant_analysis.data_models import (
    MetricSummary,
    SentenceObservation,
    ClusterSummary,
    PatternDistribution,
    ParticipantAnalysisResult
)

EPS = 1e-9


def robust_z(series: pd.Series) -> pd.Series:
    """Robust within-participant Z-score based on median and MAD."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = x.median()
    mad = (x - med).abs().median()

    if pd.isna(med):
        return pd.Series(np.nan, index=x.index, dtype=float)
    if pd.isna(mad) or mad < EPS:
        sd = x.std()
        if pd.isna(sd) or sd < EPS:
            return pd.Series(np.where(x.notna(), 0.0, np.nan), index=x.index, dtype=float)
        return (x - x.mean()) / sd

    return 0.67448975 * (x - med) / mad


class ParticipantAnalyzer:
    """Programmatic, deterministic analyzer for participant EEG and eye-tracking data."""

    def __init__(self, csv_path: Optional[Path] = None):
        if csv_path is not None:
            self.csv_path = Path(csv_path).resolve()
        else:
            self.csv_path = get_active_participant_csv()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Participant CSV not found at: {self.csv_path}")

        set_active_participant_csv(self.csv_path)

        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        self.participant_id: str = "Unknown"
        self.source_file: str = "Unknown"
        self._personal_ranges: Dict[str, Dict[str, float]] = {}
        self._cluster_interpretations: Dict[int, Dict[str, str]] = {}

        self._preprocess()
        self._load_supplemental_files()

    def _preprocess(self) -> None:
        """Ensure all required columns, ratios, and composite scores exist."""
        if "patient_id" in self.df.columns and not self.df["patient_id"].isna().all():
            self.participant_id = str(self.df["patient_id"].iloc[0])
        else:
            self.participant_id = "participant_heldout"

        if "source_file" in self.df.columns and not self.df["source_file"].isna().all():
            self.source_file = str(self.df["source_file"].iloc[0])
        else:
            self.source_file = self.csv_path.name

        if "sentence_index" not in self.df.columns:
            self.df["sentence_index"] = np.arange(len(self.df))

        # Derive power ratios if not already present
        if "Theta_Alpha_Ratio" not in self.df.columns and {"ThetaPower", "AlphaPower"}.issubset(self.df.columns):
            self.df["Theta_Alpha_Ratio"] = self.df["ThetaPower"] / (self.df["AlphaPower"].abs() + EPS)

        if "Theta_Beta_Ratio" not in self.df.columns and {"ThetaPower", "BetaPower"}.issubset(self.df.columns):
            self.df["Theta_Beta_Ratio"] = self.df["ThetaPower"] / (self.df["BetaPower"].abs() + EPS)

        # Ensure robust z-scores exist
        for m in list(METRIC_DEFINITIONS.keys()):
            if m in self.df.columns:
                self.df[m] = pd.to_numeric(self.df[m], errors="coerce")
                rz_col = f"{m}_rz"
                if rz_col not in self.df.columns:
                    self.df[rz_col] = robust_z(self.df[m])

        # If cognitive patterns are not in input CSV, calculate standard fallback
        if "cognitive_pattern" not in self.df.columns:
            self._derive_exploratory_patterns()

    def _derive_exploratory_patterns(self) -> None:
        """Derive standard pattern indicators if reading directly from test_predictions.csv."""
        # Rolling variabilities
        for metric, target in [("EEI", "EEI_local_sd"), ("FFD", "FFD_local_sd"),
                               ("TRT", "TRT_local_sd"), ("PupilSize", "Pupil_local_sd")]:
            if metric in self.df.columns and target not in self.df.columns:
                self.df[target] = self.df[metric].rolling(10, min_periods=3).std()
                self.df[f"{target}_rz"] = robust_z(self.df[target])

        # Composite Indices
        if "REI" not in self.df.columns:
            rei_cols = [c for c in ["EEI_rz", "FMT_rz", "PupilSize_rz"] if c in self.df.columns]
            self.df["REI"] = self.df[rei_cols].mean(axis=1, skipna=True)

        if "CWI" not in self.df.columns:
            cwi_cols = [c for c in ["ThetaPower_rz", "PupilSize_rz", "GD_rz", "TRT_rz"] if c in self.df.columns]
            self.df["CWI"] = self.df[cwi_cols].mean(axis=1, skipna=True)

        instability_cols = [c for c in ["EEI_local_sd_rz", "FFD_local_sd_rz", "TRT_local_sd_rz", "Pupil_local_sd_rz"] if c in self.df.columns]
        instability = self.df[instability_cols].mean(axis=1, skipna=True) if instability_cols else pd.Series(0.0, index=self.df.index)

        if "SAI" not in self.df.columns:
            self.df["SAI"] = self.df["REI"] - instability
        if "MWLI_proxy" not in self.df.columns:
            reading_var = self.df[[c for c in ["FFD_local_sd_rz", "TRT_local_sd_rz"] if c in self.df.columns]].mean(axis=1, skipna=True)
            self.df["MWLI_proxy"] = -self.df.get("EEI_rz", 0) + reading_var
        if "VSI_proxy" not in self.df.columns:
            self.df["VSI_proxy"] = -instability

        # Evidence scores
        neg_eei = -self.df.get("EEI_rz", pd.Series(0, index=self.df.index))
        neg_ffd_var = -self.df.get("FFD_local_sd_rz", pd.Series(0, index=self.df.index))
        neg_trt_var = -self.df.get("TRT_local_sd_rz", pd.Series(0, index=self.df.index))

        self.df["focus_like_evidence"] = (self.df.get("EEI_rz", 0) + neg_ffd_var + neg_trt_var) / 3.0
        self.df["fatigue_compatible_evidence"] = (self.df.get("ThetaPower_rz", 0) + self.df.get("AlphaPower_rz", 0) + neg_eei) / 3.0
        self.df["attention_instability_evidence"] = (neg_eei + self.df.get("EEI_local_sd_rz", 0) + self.df.get("FFD_local_sd_rz", 0) + self.df.get("TRT_local_sd_rz", 0)) / 4.0
        self.df["workload_evidence"] = self.df["CWI"]

        labels, strengths, top_s, sec_s = [], [], [], []
        score_cols = ["focus_like_evidence", "fatigue_compatible_evidence", "attention_instability_evidence", "workload_evidence"]

        for _, row in self.df[score_cols].iterrows():
            valid = {k: float(row[k]) for k in score_cols if pd.notna(row[k])}
            ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
            if not ranked:
                labels.append("Insufficient data")
                strengths.append("Unavailable")
                top_s.append(np.nan)
                sec_s.append(np.nan)
                continue
            top_name, top_val = ranked[0]
            sec_val = ranked[1][1] if len(ranked) > 1 else -np.inf
            top_s.append(top_val)
            sec_s.append(sec_val if np.isfinite(sec_val) else np.nan)
            if top_val < 0.50:
                label = "Mixed / indeterminate"
            elif top_val - sec_val < 0.25:
                label = "Mixed / overlapping evidence"
            else:
                label = {
                    "focus_like_evidence": "Engagement-supported / focused-like",
                    "fatigue_compatible_evidence": "Fatigue-compatible EEG pattern",
                    "attention_instability_evidence": "Attention-instability / disengagement-like",
                    "workload_evidence": "High cognitive-workload-like pattern"
                }.get(top_name, "Mixed")
            labels.append(label)
            strengths.append(
                "Stronger heuristic evidence" if top_val >= 1.0
                else "Moderate heuristic evidence" if top_val >= 0.5
                else "Weak / no clear evidence"
            )

        self.df["cognitive_pattern"] = labels
        self.df["pattern_evidence_strength"] = strengths
        self.df["top_pattern_score"] = top_s
        self.df["second_pattern_score"] = sec_s

    def _load_supplemental_files(self) -> None:
        """Load pre-generated personal reference ranges and cluster interpretation CSVs if available."""
        if PERSONAL_RANGES_CSV.exists():
            try:
                pr_df = pd.read_csv(PERSONAL_RANGES_CSV)
                for _, row in pr_df.iterrows():
                    feat = str(row["feature"])
                    self._personal_ranges[feat] = {
                        "session_median": float(row["session_median"]),
                        "lower": float(row["personal_reference_lower"]),
                        "upper": float(row["personal_reference_upper"])
                    }
            except Exception:
                pass

        if CLUSTER_INTERPRETATION_CSV.exists():
            try:
                ci_df = pd.read_csv(CLUSTER_INTERPRETATION_CSV)
                for _, row in ci_df.iterrows():
                    cid = int(row["cluster_id"])
                    self._cluster_interpretations[cid] = {
                        "descriptive_name": str(row.get("descriptive_name", f"Cluster {cid}")),
                        "interpretation_note": str(row.get("interpretation_note", "")),
                        "dominant_evidence_score": str(row.get("dominant_evidence_score", ""))
                    }
            except Exception:
                pass

    def compute_metric_summaries(self) -> Dict[str, MetricSummary]:
        """Compute complete deterministic statistics for each physiological metric."""
        summaries: Dict[str, MetricSummary] = {}

        for metric_id, meta in METRIC_DEFINITIONS.items():
            if metric_id not in self.df.columns:
                continue

            series = pd.to_numeric(self.df[metric_id], errors="coerce").dropna()
            if len(series) == 0:
                continue

            mean_val = float(series.mean())
            med_val = float(series.median())
            std_val = float(series.std()) if len(series) > 1 else 0.0
            mad_val = float((series - med_val).abs().median())
            robust_sd = 1.4826 * mad_val
            q25, q75 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr_val = q75 - q25
            min_val = float(series.min())
            max_val = float(series.max())

            # Baseline reference determination
            if metric_id in self._personal_ranges:
                baseline_med = self._personal_ranges[metric_id]["session_median"]
                ref_lower = self._personal_ranges[metric_id]["lower"]
                ref_upper = self._personal_ranges[metric_id]["upper"]
            else:
                baseline_med = med_val
                ref_lower = baseline_med - 1.5 * robust_sd
                ref_upper = baseline_med + 1.5 * robust_sd

            # Relative percentage deviation from baseline median
            if abs(baseline_med) > EPS:
                rel_dev_pct = ((mean_val - baseline_med) / abs(baseline_med)) * 100.0
            else:
                rel_dev_pct = 0.0

            # Categorize deviation
            if mean_val > ref_upper:
                dev_cat = "Elevated Above Baseline"
            elif mean_val < ref_lower:
                dev_cat = "Below Baseline"
            else:
                dev_cat = "Within Typical Range"

            # Variability Index (Coefficient of Variation)
            variability_idx = float(std_val / (abs(mean_val) + EPS))

            summaries[metric_id] = MetricSummary(
                metric_id=metric_id,
                display_name=meta["name"],
                category=meta["category"],
                unit=meta["unit"],
                mean=round(mean_val, 4),
                median=round(med_val, 4),
                std=round(std_val, 4),
                mad=round(mad_val, 4),
                robust_sd=round(robust_sd, 4),
                iqr=round(iqr_val, 4),
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
                session_baseline=round(baseline_med, 4),
                personal_ref_lower=round(ref_lower, 4),
                personal_ref_upper=round(ref_upper, 4),
                relative_deviation_pct=round(rel_dev_pct, 2),
                deviation_category=dev_cat,
                variability_index=round(variability_idx, 4),
                description=meta["description"]
            )

        return summaries

    def compute_cluster_summaries(self) -> List[ClusterSummary]:
        """Analyze participant distribution across GMM clusters."""
        if "cluster_id" not in self.df.columns:
            return []

        clusters: List[ClusterSummary] = []
        total_count = len(self.df)

        for cluster_id, group in self.df.groupby("cluster_id"):
            cid = int(cluster_id)
            cname = str(group["cluster_name"].iloc[0]) if "cluster_name" in group.columns else f"Cluster {cid}"
            cnt = len(group)
            pct = round((cnt / total_count) * 100.0, 2)
            mean_prob = float(group["cluster_probability"].mean()) if "cluster_probability" in group.columns else 1.0

            # Dominant pattern within cluster
            if "cognitive_pattern" in group.columns:
                valid_pat = group["cognitive_pattern"].dropna()
                dom_pattern = str(valid_pat.mode().iloc[0]) if not valid_pat.empty else "Undetermined"
            else:
                dom_pattern = "Undetermined"

            # Supplemental interpretation if available
            descriptive_label = (
                self._cluster_interpretations.get(cid, {}).get("descriptive_name")
                or f"Cluster {cid} ({dom_pattern})"
            )

            # Find top differentiating features relative to session median
            diff_features = []
            for m in ["EEI", "FMT", "GD", "TRT", "PupilSize", "FixationCount", "ThetaPower", "AlphaPower"]:
                if m in group.columns and m in self.df.columns:
                    cl_vals = group[m].dropna()
                    all_vals = self.df[m].dropna()
                    if not cl_vals.empty and not all_vals.empty:
                        cl_med = float(cl_vals.median())
                        all_med = float(all_vals.median())
                        if abs(all_med) > EPS:
                            diff_pct = ((cl_med - all_med) / abs(all_med)) * 100.0
                            diff_features.append({
                                "metric": m,
                                "cluster_median": round(cl_med, 3),
                                "session_median": round(all_med, 3),
                                "diff_pct": round(diff_pct, 1)
                            })

            # Sort by largest absolute difference
            diff_features.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)

            clusters.append(ClusterSummary(
                cluster_id=cid,
                cluster_name=cname,
                sentence_count=cnt,
                percentage=pct,
                mean_membership_probability=round(mean_prob, 4),
                dominant_pattern=dom_pattern,
                descriptive_label=descriptive_label,
                top_differentiating_features=diff_features[:4]
            ))

        clusters.sort(key=lambda c: c.sentence_count, reverse=True)
        return clusters

    def compute_pattern_distribution(self) -> List[PatternDistribution]:
        """Aggregate heuristic pattern frequencies."""
        if "cognitive_pattern" not in self.df.columns:
            return []

        patterns: List[PatternDistribution] = []
        total = len(self.df)
        counts = self.df["cognitive_pattern"].value_counts()

        for pat_name, count in counts.items():
            pct = round((count / total) * 100.0, 2)
            sub = self.df[self.df["cognitive_pattern"] == pat_name]
            valid_strength = sub["pattern_evidence_strength"].dropna() if "pattern_evidence_strength" in sub.columns else pd.Series(dtype=object)
            mode_strength = str(valid_strength.mode().iloc[0]) if not valid_strength.empty else "Moderate"

            patterns.append(PatternDistribution(
                pattern_name=str(pat_name),
                sentence_count=int(count),
                percentage=pct,
                evidence_level=mode_strength
            ))

        return patterns

    def compute_sentence_observations(self) -> List[SentenceObservation]:
        """Extract detailed sentence-level variations, spotting outliers and transitions."""
        observations: List[SentenceObservation] = []

        metrics_of_interest = ["EEI", "FMT", "ASM", "GD", "TRT", "FFD", "PupilSize", "FixationCount", "REI", "CWI", "SAI", "MWLI_proxy"]

        for idx, row in self.df.iterrows():
            s_idx = int(row.get("sentence_index", idx))
            s_text = str(row.get("sentence_text", f"Sentence {s_idx}"))
            c_id = int(row.get("cluster_id", 0))
            c_name = str(row.get("cluster_name", f"Cluster {c_id}"))
            c_prob = float(row.get("cluster_probability", 1.0))
            p_status = str(row.get("prediction_status", "Normal"))
            c_pattern = str(row.get("cognitive_pattern", "Mixed"))
            p_strength = str(row.get("pattern_evidence_strength", "Moderate"))
            top_s = float(row.get("top_pattern_score", 0.0))
            sec_s = float(row["second_pattern_score"]) if pd.notna(row.get("second_pattern_score")) else None

            # Metric raw values and robust z
            metric_vals: Dict[str, float] = {}
            rz_vals: Dict[str, float] = {}
            outlier_reasons: List[str] = []

            for m in metrics_of_interest:
                if m in row and pd.notna(row[m]):
                    val = float(row[m])
                    metric_vals[m] = round(val, 3)
                rz_col = f"{m}_rz"
                if rz_col in row and pd.notna(row[rz_col]):
                    rz = float(row[rz_col])
                    rz_vals[m] = round(rz, 3)
                    if abs(rz) >= 2.5:
                        outlier_reasons.append(f"{m} markedly deviated (|robust-z|={abs(rz):.2f} >= 2.5)")

            is_outlier = len(outlier_reasons) > 0

            observations.append(SentenceObservation(
                sentence_index=s_idx,
                sentence_text=s_text,
                cluster_id=c_id,
                cluster_name=c_name,
                cluster_probability=round(c_prob, 4),
                prediction_status=p_status,
                cognitive_pattern=c_pattern,
                pattern_evidence_strength=p_strength,
                top_pattern_score=round(top_s, 3),
                second_pattern_score=round(sec_s, 3) if sec_s is not None else None,
                metrics=metric_vals,
                robust_z_scores=rz_vals,
                is_outlier=is_outlier,
                outlier_reasons=outlier_reasons
            ))

        return observations

    def generate_session_observations(self, metrics: Dict[str, MetricSummary], clusters: List[ClusterSummary], patterns: List[PatternDistribution]) -> List[str]:
        """Synthesize programmatic, purely physiological summary observations without diagnostic claims."""
        obs = []

        total = len(self.df)
        obs.append(f"Recorded participant session contains {total} sentences analyzed using non-invasive EEG and eyetracking.")

        # Eye-tracking observations
        if "TRT" in metrics and "GD" in metrics:
            trt = metrics["TRT"]
            gd = metrics["GD"]
            obs.append(
                f"Reading times: Median gaze duration was {gd.median} ms (IQR: {gd.iqr} ms), with median total reading time of {trt.median} ms. "
                f"Relative deviation from personal session baseline is {trt.relative_deviation_pct:+.1f}% ({trt.deviation_category.lower()})."
            )

        if "PupilSize" in metrics:
            pupil = metrics["PupilSize"]
            obs.append(
                f"Pupillometry: Mean pupil diameter was {pupil.mean} units with a standard deviation of {pupil.std} units "
                f"(coefficient of variation = {pupil.variability_index:.2f})."
            )

        # Spectral EEG observations
        if "EEI" in metrics and "FMT" in metrics:
            eei = metrics["EEI"]
            fmt = metrics["FMT"]
            obs.append(
                f"Spectral EEG Dynamics: Frontal Midline Theta (FMT) baseline median was {fmt.session_baseline} µV². "
                f"EEG Engagement Index (EEI = Beta/(Alpha+Theta)) averaged {eei.mean:.3f} across the reading session."
            )

        # Dominant cluster
        if clusters:
            top_c = clusters[0]
            obs.append(
                f"Unsupervised GMM Structure: Participant sentences predominantly grouped into {top_c.cluster_name} "
                f"({top_c.percentage}% of sentences, mean cluster confidence {top_c.mean_membership_probability:.2f}), "
                f"characterized by: {top_c.descriptive_label}."
            )

        # Dominant patterns
        if patterns:
            top_p = patterns[0]
            obs.append(
                f"Session Pattern Evidence: The most frequent heuristic classification was '{top_p.pattern_name}' "
                f"covering {top_p.percentage}% of read sentences ({top_p.sentence_count}/{total})."
            )

        return obs

    def analyze(self) -> ParticipantAnalysisResult:
        """Execute full deterministic pipeline and return structured result."""
        metric_summaries = self.compute_metric_summaries()
        cluster_summaries = self.compute_cluster_summaries()
        pattern_dist = self.compute_pattern_distribution()
        sentence_obs = self.compute_sentence_observations()
        session_obs = self.generate_session_observations(metric_summaries, cluster_summaries, pattern_dist)

        # Student Cognitive Learning analysis: Flow vs Breakdown & Fatigue Curve
        breakdown_rep = None
        learning_eff = None
        fatigue_curve = None
        try:
            from neuroread_rag.participant_analysis.sentence_comparator import SentenceComparator
            comp = SentenceComparator(self.df)
            comp_res = comp.run()
            breakdown_rep = comp_res.get("report")
            if breakdown_rep:
                learning_eff = breakdown_rep.get("learning_efficiency")
                fatigue_curve = breakdown_rep.get("session_fatigue_curve")
        except Exception:
            pass

        return ParticipantAnalysisResult(
            participant_id=self.participant_id,
            source_file=self.source_file,
            total_sentences=len(self.df),
            metrics=metric_summaries,
            clusters=cluster_summaries,
            patterns=pattern_dist,
            sentence_observations=sentence_obs,
            session_level_observations=session_obs,
            disclaimers=MANDATORY_DISCLAIMERS,
            breakdown_report=breakdown_rep,
            learning_efficiency=learning_eff,
            session_fatigue_curve=fatigue_curve,
        )

    def save_analysis(self, output_path: Optional[Path] = None) -> Path:
        """Save analysis to JSON file."""
        result = self.analyze()
        out_file = output_path or (OUTPUT_DIR / f"{self.participant_id}_analysis.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return out_file
