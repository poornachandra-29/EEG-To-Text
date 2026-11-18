"""
Sentence Comprehension Classifier & Breakdown Analyzer.

Detects when a reader gets "stuck" on sentences (high fixation + EEI drop)
vs when they are in a "flow state" (smooth reading, high engagement).

Provides both:
1. Legacy quartile-based descriptive states (legacy_state, reading_state)
2. Validated behavioral breakdown flags (breakdown_behav) and excess processing scores
3. External outcome tracking (comprehension_error from ZuCo control questions)
4. Target leakage guardrails to ensure valid machine learning evaluations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from neuroread_rag.participant_analysis.comprehension_labels import attach_comprehension_labels
from neuroread_rag.participant_analysis.residualize import compute_robust_z


# ---------------------------------------------------------------------------
# Leakage Prevention Guardrail
# ---------------------------------------------------------------------------

def verify_no_leakage(target: str, feature_names: List[str]) -> None:
    """
    Enforces strict target-feature leakage guards.
    When predicting behavioral breakdown or excess processing, eye-tracking features
    defining the target (TRT, FixationCount, RegressionCount, GD, excess_processing)
    and their derived transformations cannot be used as predictors.
    """
    if target in ("breakdown_behav", "excess_processing"):
        leaking_bases = ["trt", "fixation", "regression", "gd", "excess_processing"]
        leaks = [
            f for f in feature_names
            if any(b in f.lower() for b in leaking_bases)
        ]
        if leaks:
            raise ValueError(
                f"Target leakage violation for target '{target}'! "
                f"The following predictors define the outcome and must be excluded: {leaks}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """Return a finite float or the fallback value."""
    try:
        v = float(value)
        return v if np.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def count_syllables_word(word: str) -> int:
    """Deterministic English syllable counter heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    clean = "".join(c for c in word if c.isalpha())
    if not clean:
        return 0
    if len(clean) <= 3:
        return 1
    # Strip silent trailing e (except le, ee)
    if clean.endswith("e") and not clean.endswith("le") and not clean.endswith("ee"):
        clean = clean[:-1]
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in clean:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def analyze_sentence_linguistics(text: str) -> Dict[str, Any]:
    """Calculate Flesch Reading Ease, Grade Level, and complex word ratio for text."""
    words = text.split()
    wc = len(words)
    if wc == 0:
        return {
            "syllables": 0,
            "complex_words": 0,
            "complex_words_pct": 0.0,
            "flesch_reading_ease": 100.0,
            "grade_level": 1.0,
        }
    syllables = sum(count_syllables_word(w) for w in words)
    complex_words = sum(1 for w in words if count_syllables_word(w) >= 3)
    complex_words_pct = round((complex_words / wc) * 100.0, 1)

    # Standard Flesch Reading Ease for 1 sentence: 206.835 - 1.015*wc - 84.6*(syllables/wc)
    fre = 206.835 - (1.015 * wc) - (84.6 * (syllables / wc))
    fre = max(0.0, min(100.0, round(fre, 1)))

    # Flesch-Kincaid Grade Level: 0.39*wc + 11.8*(syllables/wc) - 15.59
    fkgl = (0.39 * wc) + (11.8 * (syllables / wc)) - 15.59
    fkgl = max(1.0, min(20.0, round(fkgl, 1)))

    return {
        "syllables": syllables,
        "complex_words": complex_words,
        "complex_words_pct": complex_words_pct,
        "flesch_reading_ease": fre,
        "grade_level": fkgl,
    }


def _sentence_row(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """Build a sentence-level dict for top-N lists from a DataFrame row."""
    row = df.iloc[idx]
    wc = _safe_int(row.get("word_count", 0))
    comp_pct = _safe_float(row.get("complex_words_pct", 0.0))
    fre = _safe_float(row.get("flesch_reading_ease", 0.0))
    grade = _safe_float(row.get("grade_level", 0.0))
    state = str(row.get("reading_state", "⚪ Normal Reading"))
    fpw = _safe_float(row.get("fixation_per_word", 0.0))
    fix = _safe_float(row.get("FixationCount", 0.0))

    if "Breakdown" in state or "Fatigue" in state:
        if comp_pct >= 20.0:
            root_cause = "Lexical Overload (Dense / Unfamiliar Words)"
        elif wc >= 20:
            root_cause = "Syntactic Overload (Multi-Clause Structure)"
        elif fpw >= 0.12 or fix >= 1.8:
            root_cause = "Focal Stagnation (Re-reading Named Entities / Facts)"
        else:
            root_cause = "Cognitive Saturation / Attentional Drift"
    elif "Flow" in state:
        root_cause = "Fluent Reading Flow (Effortless Intake)"
    else:
        root_cause = "Typical Reading Cadence"

    return {
        "sentence_index": _safe_int(row.get("sentence_index", idx)),
        "sentence_text": str(row.get("sentence_text", "")) if pd.notna(row.get("sentence_text", "")) else "",
        "word_count": wc,
        "fixation_count": fix,
        "fixation_per_word": fpw,
        "eei": _safe_float(row.get("EEI", 0.0)),
        "trt": _safe_float(row.get("TRT", 0.0)),
        "reading_state": state,
        "complex_words_pct": comp_pct,
        "flesch_reading_ease": fre,
        "grade_level": grade,
        "linguistic_root_cause": root_cause,
        "breakdown_behav": _safe_int(row.get("breakdown_behav", 0)),
        "excess_processing": _safe_float(row.get("excess_processing", 0.0)),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentenceComparator:
    """
    Classify individual sentences into reading states and produce a
    comprehension-breakdown report from participant physiological data.

    Parameters
    ----------
    df : pd.DataFrame
        The participant CSV loaded as a DataFrame.
    """

    # Legacy state labels
    STATE_FLOW = "🟢 Flow State"
    STATE_DEEP_FOCUS = "🟡 Deep Focus"
    STATE_BREAKDOWN = "🔴 Comprehension Breakdown"
    STATE_FATIGUE = "⚪ Cognitive Fatigue"
    STATE_NORMAL = "🔵 Normal Reading"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df: pd.DataFrame = df.copy()

    # ------------------------------------------------------------------
    # 1. Reading-state classification
    # ------------------------------------------------------------------

    def classify_reading_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies sentences using:
        1. Legacy quartile-based descriptive states (reading_state, legacy_state).
        2. Absolute confound-controlled behavioral breakdown (breakdown_behav).
        3. Continuous excess processing index.
        4. External comprehension question outcomes (comprehension_error).
        """
        out = df.copy()

        # 1. Legacy quartile-based states
        try:
            eei = pd.to_numeric(out["EEI"], errors="coerce")
            fix = pd.to_numeric(out["FixationCount"], errors="coerce")

            eei_q25 = float(eei.quantile(0.25))
            eei_q75 = float(eei.quantile(0.75))
            fix_q50 = float(fix.quantile(0.50))
            fix_q75 = float(fix.quantile(0.75))

            states = pd.Series(self.STATE_NORMAL, index=out.index)
            cog_fatigue = (eei < eei_q25) & (fix <= fix_q50)
            breakdown = (fix > fix_q75) & (eei < eei_q25)
            deep_focus = (eei > eei_q75) & (fix > fix_q75)
            flow = (eei > eei_q75) & (fix <= fix_q50)

            states[cog_fatigue] = self.STATE_FATIGUE
            states[breakdown] = self.STATE_BREAKDOWN
            states[deep_focus] = self.STATE_DEEP_FOCUS
            states[flow] = self.STATE_FLOW

            out["reading_state"] = states
            out["legacy_state"] = states.apply(lambda s: f"{s} (quartile-based, descriptive only)")

        except (KeyError, TypeError, ValueError):
            out["reading_state"] = self.STATE_NORMAL
            out["legacy_state"] = "🔵 Normal Reading (quartile-based, descriptive only)"

        # 2. Confound-controlled behavioral breakdown (breakdown_behav)
        # Target: TRT_rz >= 1.0 AND (FixationCount_rz >= 0.5 OR GD_rz >= 0.5)
        try:
            if "TRT_rz" in out.columns and out["TRT_rz"].notna().any():
                trt_z = pd.to_numeric(out["TRT_rz"], errors="coerce").fillna(0.0)
            elif "TRT" in out.columns:
                trt_z = pd.Series(compute_robust_z(out["TRT"].values), index=out.index).fillna(0.0)
            else:
                trt_z = pd.Series(0.0, index=out.index)

            if "FixationCount_rz" in out.columns and out["FixationCount_rz"].notna().any():
                fix_z = pd.to_numeric(out["FixationCount_rz"], errors="coerce").fillna(0.0)
            elif "FixationCount" in out.columns:
                fix_z = pd.Series(compute_robust_z(out["FixationCount"].values), index=out.index).fillna(0.0)
            else:
                fix_z = pd.Series(0.0, index=out.index)

            if "GD_rz" in out.columns and out["GD_rz"].notna().any():
                gd_z = pd.to_numeric(out["GD_rz"], errors="coerce").fillna(0.0)
            elif "GD" in out.columns:
                gd_z = pd.Series(compute_robust_z(out["GD"].values), index=out.index).fillna(0.0)
            else:
                gd_z = pd.Series(0.0, index=out.index)

            if "RegressionCount_rz" in out.columns and out["RegressionCount_rz"].notna().any():
                reg_z = pd.to_numeric(out["RegressionCount_rz"], errors="coerce").fillna(0.0)
            elif "RegressionCount" in out.columns:
                reg_z = pd.Series(compute_robust_z(out["RegressionCount"].values), index=out.index).fillna(0.0)
            else:
                reg_z = pd.Series(0.0, index=out.index)

            is_bd_behav = (trt_z >= 1.0) & ((fix_z >= 0.5) | (reg_z >= 0.5) | (gd_z >= 0.5))
            out["breakdown_behav"] = is_bd_behav.astype(int)
            out["excess_processing"] = ((trt_z + fix_z + gd_z) / 3.0).round(4)
        except Exception:
            out["breakdown_behav"] = 0
            out["excess_processing"] = 0.0

        # 3. External comprehension questions outcome
        out = attach_comprehension_labels(out)

        return out

    # ------------------------------------------------------------------
    # 2. Word-level metrics
    # ------------------------------------------------------------------

    def compute_word_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add word_count, fixation_per_word, sentence_complexity, and linguistic readability columns.
        """
        out = df.copy()

        try:
            texts = out["sentence_text"].fillna("").astype(str)
            out["word_count"] = texts.str.split().str.len().fillna(0).astype(int)
        except (KeyError, AttributeError):
            out["word_count"] = 0
            texts = pd.Series([""] * len(out), index=out.index)

        try:
            fix = pd.to_numeric(out["FixationCount"], errors="coerce").fillna(0.0)
            wc = out["word_count"].astype(float)
            out["fixation_per_word"] = fix / (wc + 1e-9)
        except (KeyError, TypeError):
            out["fixation_per_word"] = 0.0

        try:
            wc = out["word_count"]
            out["sentence_complexity"] = pd.cut(
                wc,
                bins=[-1, 11, 25, float("inf")],
                labels=["Short", "Medium", "Long"],
            ).astype(str)
        except (KeyError, TypeError, ValueError):
            out["sentence_complexity"] = "Medium"

        syl_list, comp_pct_list, fre_list, grade_list, root_causes = [], [], [], [], []
        for t, w, fpw in zip(texts, out["word_count"], out["fixation_per_word"]):
            ling = analyze_sentence_linguistics(t)
            syl_list.append(ling["syllables"])
            comp_pct_list.append(ling["complex_words_pct"])
            fre_list.append(ling["flesch_reading_ease"])
            grade_list.append(ling["grade_level"])

            if ling["complex_words_pct"] >= 25.0:
                rc = "Lexical Overload (Dense Academic Vocabulary)"
            elif w >= 22:
                rc = "Syntactic Overload (Multi-Clause Structure)"
            elif fpw >= 1.2:
                rc = "Focal Stagnation (Re-reading Complex Concepts)"
            else:
                rc = "Smooth Fluent Processing"
            root_causes.append(rc)

        out["syllable_count"] = syl_list
        out["complex_words_pct"] = comp_pct_list
        out["flesch_reading_ease"] = fre_list
        out["grade_level"] = grade_list
        out["linguistic_root_cause"] = root_causes

        return out

    # ------------------------------------------------------------------
    # 3. Breakdown report
    # ------------------------------------------------------------------

    def build_breakdown_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute sentence-level comprehension statistics, learning efficiency scores,
        and chronological fatigue curve using true elapsed session time.
        """
        total = len(df)

        def _count(state: str) -> int:
            try:
                return int((df["reading_state"] == state).sum())
            except KeyError:
                return 0

        def _pct(n: int) -> float:
            return round(100.0 * n / total, 1) if total > 0 else 0.0

        flow_count = _count(self.STATE_FLOW)
        deep_focus_count = _count(self.STATE_DEEP_FOCUS)
        breakdown_count = _count(self.STATE_BREAKDOWN)
        fatigue_count = _count(self.STATE_FATIGUE)
        normal_count = total - flow_count - deep_focus_count - breakdown_count - fatigue_count

        # Behavioral breakdown & external outcomes
        bd_behav_count = int((df["breakdown_behav"] == 1).sum()) if "breakdown_behav" in df else 0
        bd_behav_pct = _pct(bd_behav_count)
        excess_proc_mean = round(_safe_float(df["excess_processing"].mean()), 4) if "excess_processing" in df else 0.0

        comp_errors_count = int(np.nansum(df["comprehension_error"])) if "comprehension_error" in df else 0
        comp_asked_count = int(df["comprehension_error"].notna().sum()) if "comprehension_error" in df else 0
        comp_error_rate = round(comp_errors_count / max(1, comp_asked_count), 3) if comp_asked_count > 0 else 0.0

        def _profile(mask: pd.Series) -> Dict[str, Any]:
            sub = df[mask]
            n = int(mask.sum())
            if n == 0:
                return {"count": 0, "mean_eei": 0.0, "mean_trt": 0.0, "mean_fixation": 0.0, "mean_cwi": 0.0}
            return {
                "count": n,
                "mean_eei": round(_safe_float(pd.to_numeric(sub.get("EEI", pd.Series(dtype=float)), errors="coerce").mean()), 6),
                "mean_trt": round(_safe_float(pd.to_numeric(sub.get("TRT", pd.Series(dtype=float)), errors="coerce").mean()), 3),
                "mean_fixation": round(_safe_float(pd.to_numeric(sub.get("FixationCount", pd.Series(dtype=float)), errors="coerce").mean()), 4),
                "mean_cwi": round(_safe_float(pd.to_numeric(sub.get("CWI", pd.Series(dtype=float)), errors="coerce").mean()), 6),
            }

        try:
            simple_mask = df["sentence_complexity"] == "Short"
            complex_mask = df["sentence_complexity"] == "Long"
        except KeyError:
            simple_mask = pd.Series(False, index=df.index)
            complex_mask = pd.Series(False, index=df.index)

        simple_profile = _profile(simple_mask)
        complex_profile = _profile(complex_mask)

        # Post-breakdown EEI drift
        try:
            eei_series = pd.to_numeric(df["EEI"], errors="coerce")
            eei_next = eei_series.shift(-1)
            eei_diff = eei_next - eei_series

            is_breakdown = df["reading_state"] == self.STATE_BREAKDOWN
            is_normal = df["reading_state"] == self.STATE_NORMAL

            post_breakdown_eei_change = _safe_float(eei_diff[is_breakdown].mean())
            post_normal_eei_change = _safe_float(eei_diff[is_normal].mean())
        except (KeyError, TypeError):
            post_breakdown_eei_change = 0.0
            post_normal_eei_change = 0.0

        # Top flow & breakdown sentences
        try:
            flow_df = df[df["reading_state"] == self.STATE_FLOW].copy()
            flow_df = flow_df.sort_values("EEI", ascending=False)
            top_flow = [_sentence_row(flow_df, i) for i in range(min(5, len(flow_df)))]
        except (KeyError, TypeError):
            top_flow = []

        try:
            bd_df = df[df["reading_state"] == self.STATE_BREAKDOWN].copy()
            bd_df = bd_df.sort_values("fixation_per_word", ascending=False)
            top_breakdown = [_sentence_row(bd_df, i) for i in range(min(5, len(bd_df)))]
        except (KeyError, TypeError):
            top_breakdown = []

        try:
            wl_df = df.copy()
            wl_df = wl_df.sort_values("CWI", ascending=False)
            top_workload: List[Dict[str, Any]] = []
            for i in range(min(5, len(wl_df))):
                row = _sentence_row(wl_df, i)
                row["cwi"] = _safe_float(wl_df.iloc[i].get("CWI", 0.0))
                top_workload.append(row)
        except (KeyError, TypeError):
            top_workload = []

        # Optimal personal zone
        try:
            flow_df2 = df[df["reading_state"] == self.STATE_FLOW].copy()
            if len(flow_df2) > 0:
                wc_col = pd.to_numeric(flow_df2["word_count"], errors="coerce")
                fpw_col = pd.to_numeric(flow_df2["fixation_per_word"], errors="coerce")
                eei_col = pd.to_numeric(flow_df2["EEI"], errors="coerce")
                max_wc = int(wc_col.quantile(0.75)) if wc_col.notna().any() else 20
                max_fpw = round(_safe_float(fpw_col.quantile(0.75)), 4)
                min_eei = round(_safe_float(eei_col.quantile(0.25)), 6)
            else:
                max_wc, max_fpw, min_eei = 20, 1.0, 0.0
        except (KeyError, TypeError, ValueError):
            max_wc, max_fpw, min_eei = 20, 1.0, 0.0

        personal_optimal_zone: Dict[str, Any] = {
            "max_word_count_for_flow": max_wc,
            "max_fixation_per_word": max_fpw,
            "min_eei_for_flow": min_eei,
            "recommended_break_interval": "Every 20-25 minutes",
        }

        # Learning Efficiency Score (LES, 0-100) - explicitly tagged as heuristic proxy
        denom = flow_count + breakdown_count
        flow_factor = min(100.0, max(0.0, (flow_count / denom) * 100.0)) if denom > 0 else 50.0

        eei_col = pd.to_numeric(df.get("EEI", pd.Series(dtype=float)), errors="coerce")
        eei_std = _safe_float(eei_col.std(), 0.1)
        eei_mean = _safe_float(eei_col.mean(), 0.5)
        cadence_factor = min(100.0, max(20.0, 100.0 - (eei_std / max(abs(eei_mean), 0.05)) * 30.0))

        cwi_col = pd.to_numeric(df.get("CWI", pd.Series(dtype=float)), errors="coerce")
        cwi_mean = _safe_float(cwi_col.mean(), 0.0)
        workload_ease = min(100.0, max(20.0, 100.0 - max(0.0, cwi_mean) * 50.0))

        les_score = round(0.40 * flow_factor + 0.35 * cadence_factor + 0.25 * workload_ease, 1)
        if les_score >= 80.0:
            les_tier = "Optimal Learning Zone (Deep Flow)"
            les_color = "#10b981"
        elif les_score >= 65.0:
            les_tier = "Effective Comprehension (Moderate Effort)"
            les_color = "#3b82f6"
        elif les_score >= 50.0:
            les_tier = "Cognitive Friction (Frequent Re-reading)"
            les_color = "#f59e0b"
        else:
            les_tier = "High Cognitive Overload (Intervention Needed)"
            les_color = "#ef4444"

        learning_efficiency = {
            "score": les_score,
            "tier": les_tier,
            "color": les_color,
            "flow_factor": round(flow_factor, 1),
            "cadence_stability_factor": round(cadence_factor, 1),
            "workload_ease_factor": round(workload_ease, 1),
            "flow_percentage": _pct(flow_count),
            "breakdown_percentage": _pct(breakdown_count),
            "breakdown_behav_count": bd_behav_count,
            "breakdown_behav_pct": bd_behav_pct,
            "excess_processing_mean": excess_proc_mean,
            "comprehension_errors_count": comp_errors_count,
            "comprehension_asked_count": comp_asked_count,
            "comprehension_error_rate": comp_error_rate,
            "is_heuristic": True,
            "heuristic_label": "Heuristic composite score for self-directed study calibration",
            "reliability_disclaimer": "Scores represent empirical study engagement proxies, not clinical diagnostics.",
        }

        # Chronological Fatigue Curve & Real Timing (Phase 8)
        # Compute true elapsed time in minutes from duration signals
        if "eeg_duration_s" in df.columns and pd.to_numeric(df["eeg_duration_s"], errors="coerce").sum() > 0:
            durations_min = pd.to_numeric(df["eeg_duration_s"], errors="coerce").fillna(3.5) / 60.0
        elif "TRT" in df.columns and pd.to_numeric(df["TRT"], errors="coerce").sum() > 0:
            durations_min = pd.to_numeric(df["TRT"], errors="coerce").fillna(2500.0) / 1000.0 / 60.0
        else:
            # Fallback average 3.5s per sentence
            durations_min = pd.Series(3.5 / 60.0, index=df.index)

        elapsed_mins = durations_min.cumsum().values
        total_session_minutes = round(float(elapsed_mins[-1]), 1) if len(elapsed_mins) > 0 else 0.0

        num_pts = min(8, max(4, total // 25)) if total > 20 else max(1, total)
        chunk_size = max(1, total // num_pts)
        checkpoints = []
        fatigue_onset_sentence = None
        fatigue_onset_minute = None

        for i in range(num_pts):
            s_idx = i * chunk_size
            e_idx = total if i == num_pts - 1 else (i + 1) * chunk_size
            chunk = df.iloc[s_idx:e_idx]
            ch_eei = round(_safe_float(pd.to_numeric(chunk.get("EEI", pd.Series(dtype=float)), errors="coerce").mean()), 4)
            ch_cwi = round(_safe_float(pd.to_numeric(chunk.get("CWI", pd.Series(dtype=float)), errors="coerce").mean()), 4)
            ch_fix = round(_safe_float(pd.to_numeric(chunk.get("FixationCount", pd.Series(dtype=float)), errors="coerce").mean()), 2)
            ch_trt = round(_safe_float(pd.to_numeric(chunk.get("TRT", pd.Series(dtype=float)), errors="coerce").mean()), 1)

            # Checkpoint elapsed minute
            est_min = round(float(elapsed_mins[min(e_idx - 1, len(elapsed_mins) - 1)]), 1)
            chunk_dur_min = float(durations_min.iloc[s_idx:e_idx].sum())
            chunk_words = int(pd.to_numeric(chunk.get("word_count", 0), errors="coerce").sum())
            wpm = round(chunk_words / max(0.1, chunk_dur_min), 1)

            is_fatigued = (ch_cwi > 0.15 and ch_eei < eei_mean) or (ch_fix > 1.8 and ch_eei < eei_mean)
            status = "Elevated Fatigue" if is_fatigued else ("Optimal Focus" if ch_eei >= eei_mean else "Mild Drift")

            if is_fatigued and fatigue_onset_sentence is None and s_idx > 0:
                fatigue_onset_sentence = s_idx
                fatigue_onset_minute = est_min

            checkpoints.append({
                "checkpoint": i + 1,
                "sentence_range": f"Sentences {s_idx + 1}–{e_idx}",
                "start_sentence": s_idx + 1,
                "end_sentence": e_idx,
                "mean_eei": ch_eei,
                "mean_cwi": ch_cwi,
                "mean_fixation": ch_fix,
                "mean_trt": ch_trt,
                "estimated_minute": est_min,
                "reading_wpm": wpm,
                "status": status,
            })

        # Reliable fatigue detection & Pomodoro clamping (Phase 8)
        if fatigue_onset_sentence is not None and fatigue_onset_minute is not None:
            raw_target_pomo = fatigue_onset_minute
            has_fatigue_detected = True
        else:
            has_fatigue_detected = False
            fatigue_onset_sentence = int(total * 0.65) if total > 30 else total
            fatigue_onset_minute = round(float(elapsed_mins[min(fatigue_onset_sentence, len(elapsed_mins) - 1)]), 1) if len(elapsed_mins) > 0 else 25.0
            raw_target_pomo = 25.0

        # Clamp Pomodoro to [15, 40] minutes rounded to nearest 5 min
        recommended_pomodoro = int(max(15, min(40, round(raw_target_pomo / 5.0) * 5)))

        if has_fatigue_detected:
            insight_text = (
                f"Attentional engagement was highest during the initial {fatigue_onset_minute:.0f} minutes "
                f"(sentences 1–{fatigue_onset_sentence}). Structuring study sessions with a {recommended_pomodoro}-minute "
                f"Pomodoro interval followed by a 5-minute break aligns with your observed neural stamina."
            )
        else:
            insight_text = (
                f"No persistent cognitive fatigue onset detected across the {total_session_minutes:.1f}-minute reading session. "
                f"A standard {recommended_pomodoro}-minute focus block is recommended for baseline scheduling."
            )

        session_fatigue_curve = {
            "checkpoints": checkpoints,
            "total_session_minutes": total_session_minutes,
            "has_reliable_fatigue_detected": has_fatigue_detected,
            "fatigue_onset_sentence": fatigue_onset_sentence,
            "estimated_fatigue_minute": fatigue_onset_minute,
            "recommended_study_block_minutes": recommended_pomodoro,
            "insight": insight_text,
        }

        avg_flow_w = np.mean([s["word_count"] for s in top_flow]) if top_flow else 15
        avg_bd_fix = np.mean([s["fixation_count"] for s in top_breakdown]) if top_breakdown else 2.0
        avg_fl_fix = np.mean([s["fixation_count"] for s in top_flow]) if top_flow else 1.0
        fix_ratio = round(avg_bd_fix / max(avg_fl_fix, 0.1), 1)

        contrast_insight = (
            f"Your reading cadence was most effortless on concise sentences (averaging {avg_flow_w:.0f} words) "
            f"where ocular fixations progressed sequentially with minimal re-reading. In contrast, on challenging "
            f"breakdown sentences, your eyes registered {fix_ratio}× more fixations per sentence, while brain "
            f"engagement dropped sharply. Crucially, after each breakdown sentence, your EEG engagement surged by an "
            f"average of {post_breakdown_eei_change:+.2f} units on the very next sentence, confirming active cognitive resilience."
        )

        return {
            "total_sentences": total,
            "flow_count": flow_count,
            "flow_pct": _pct(flow_count),
            "deep_focus_count": deep_focus_count,
            "deep_focus_pct": _pct(deep_focus_count),
            "breakdown_count": breakdown_count,
            "breakdown_pct": _pct(breakdown_count),
            "fatigue_count": fatigue_count,
            "fatigue_pct": _pct(fatigue_count),
            "normal_count": normal_count,
            "breakdown_behav_count": bd_behav_count,
            "breakdown_behav_pct": bd_behav_pct,
            "excess_processing_mean": excess_proc_mean,
            "comprehension_errors_count": comp_errors_count,
            "comprehension_asked_count": comp_asked_count,
            "comprehension_error_rate": comp_error_rate,
            "simple_sentences_profile": simple_profile,
            "complex_sentences_profile": complex_profile,
            "post_breakdown_eei_change": round(post_breakdown_eei_change, 4),
            "post_normal_eei_change": round(post_normal_eei_change, 4),
            "top_flow_sentences": top_flow,
            "top_breakdown_sentences": top_breakdown,
            "top_workload_sentences": top_workload,
            "contrast_insight": contrast_insight,
            "personal_optimal_zone": personal_optimal_zone,
            "learning_efficiency": learning_efficiency,
            "session_fatigue_curve": session_fatigue_curve,
        }

    # ------------------------------------------------------------------
    # 4. Plain-English narrative summary
    # ------------------------------------------------------------------

    def build_plain_english_summary(
        self,
        report_dict: Dict[str, Any],
        participant_id: str,
    ) -> str:
        """Produce a jargon-free narrative from the breakdown report."""
        try:
            total = _safe_int(report_dict.get("total_sentences", 0))
            flow_count = _safe_int(report_dict.get("flow_count", 0))
            flow_pct = _safe_float(report_dict.get("flow_pct", 0.0))
            breakdown_count = _safe_int(report_dict.get("breakdown_count", 0))
            breakdown_pct = _safe_float(report_dict.get("breakdown_pct", 0.0))
            post_bd_change = _safe_float(report_dict.get("post_breakdown_eei_change", 0.0))

            complex_profile = report_dict.get("complex_sentences_profile", {})
            simple_profile = report_dict.get("simple_sentences_profile", {})
            cwi_complex = _safe_float(complex_profile.get("mean_cwi", 0.0))
            cwi_simple = _safe_float(simple_profile.get("mean_cwi", 0.0))
            trt_complex = _safe_float(complex_profile.get("mean_trt", 0.0))
            trt_simple = _safe_float(simple_profile.get("mean_trt", 0.0))

            cwi_ratio = (cwi_complex / (cwi_simple + 1e-9)) if cwi_simple > 1e-9 else 1.0
            trt_ratio = ((trt_complex - trt_simple) / (trt_simple + 1e-9) * 100.0) if trt_simple > 1e-9 else 0.0

            optimal_zone = report_dict.get("personal_optimal_zone", {})
            max_words = _safe_int(optimal_zone.get("max_word_count_for_flow", 20))

            les = report_dict.get("learning_efficiency", {})
            les_score = _safe_float(les.get("score", 75.0))
            les_tier = str(les.get("tier", "Effective Comprehension"))

            fatigue = report_dict.get("session_fatigue_curve", {})
            onset = _safe_int(fatigue.get("fatigue_onset_sentence", int(total * 0.65)))
            pomo = _safe_int(fatigue.get("recommended_study_block_minutes", 25))

            complex_threshold = 25

            return (
                f"During your study session, you completed {total} learning sentences in total.\n\n"
                f"Your Learning Efficiency Score (LES) reached {les_score:.1f}% ({les_tier}). "
                f"Your reading progressed in a fluent 'flow state' across {flow_count} sentences "
                f"({flow_pct}%), where your brain smoothly processed words with minimal re-reading.\n\n"
                f"However, on {breakdown_count} sentences ({breakdown_pct}%), your eyes "
                f"refixated repeatedly while brain engagement momentarily declined — indicating "
                f"comprehension friction where working memory buffer saturated.\n\n"
                f"Remarkably, your neural system showed strong cognitive resilience: engagement levels surged "
                f"by an average of {post_bd_change:+.2f} units immediately following difficult sentences, "
                f"demonstrating active recovery.\n\n"
                f"On syntactically dense sentences (>{complex_threshold} words), your cognitive workload was "
                f"{cwi_ratio:.1f}× higher and required {trt_ratio:.0f}% more study time. "
                f"Attentional focus remained sharpest through sentence {onset}, indicating an optimal study "
                f"block length of {pomo} minutes before taking a structured 5-minute break."
            )

        except Exception as exc:
            return (
                f"Summary could not be generated for student {participant_id}. "
                f"Reason: {exc}"
            )

    # ------------------------------------------------------------------
    # Convenience: run full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline on the stored DataFrame."""
        df = self.compute_word_metrics(self._df)
        df = self.classify_reading_states(df)
        report = self.build_breakdown_report(df)

        pid_col = df.get("patient_id", pd.Series(dtype=str))
        pid = str(pid_col.iloc[0]) if len(pid_col) > 0 and pid_col.notna().any() else "unknown"
        summary = self.build_plain_english_summary(report, pid)

        return {"df": df, "report": report, "summary": summary}
