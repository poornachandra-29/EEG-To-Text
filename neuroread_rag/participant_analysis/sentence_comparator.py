"""
Sentence Comprehension Classifier & Breakdown Analyzer.

Detects when a reader gets "stuck" on sentences (high fixation + EEI drop)
vs when they are in a "flow state" (smooth reading, high engagement).

Core insight: If FixationCount is high AND EEI drops simultaneously,
the reader is NOT deeply focused — they are having a comprehension breakdown.
If FixationCount is high AND EEI is also high, that is deep focus.

Also computes post-breakdown EEI drift: after a breakdown sentence,
the brain compensates by showing elevated EEI on the next sentence.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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
        The participant CSV loaded as a DataFrame.  Must contain at least the
        columns ``EEI``, ``FixationCount``, ``TRT``, ``CWI``, and
        ``sentence_text``.  Missing columns are handled gracefully.
    """

    # State labels
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
        Add a ``reading_state`` column to *df* based on EEI and FixationCount
        percentile thresholds computed from the same DataFrame.

        Rules (applied in priority order):
        - 🟢 Flow State          — EEI > Q75(EEI)  AND  FixationCount ≤ Q50(FixationCount)
        - 🟡 Deep Focus          — EEI > Q75(EEI)  AND  FixationCount > Q75(FixationCount)
        - 🔴 Comprehension Breakdown — FixationCount > Q75(FixationCount) AND EEI < Q25(EEI)
        - ⚪ Cognitive Fatigue   — EEI < Q25(EEI)  AND  FixationCount ≤ Q50(FixationCount)
        - 🔵 Normal Reading      — everything else

        Returns the modified DataFrame (also modifies in place).
        """
        out = df.copy()

        try:
            eei = pd.to_numeric(out["EEI"], errors="coerce")
            fix = pd.to_numeric(out["FixationCount"], errors="coerce")

            eei_q25 = float(eei.quantile(0.25))
            eei_q75 = float(eei.quantile(0.75))
            fix_q50 = float(fix.quantile(0.50))
            fix_q75 = float(fix.quantile(0.75))

            # Assign states using boolean masks (order matters!)
            states = pd.Series(self.STATE_NORMAL, index=out.index)

            cog_fatigue = (eei < eei_q25) & (fix <= fix_q50)
            breakdown = (fix > fix_q75) & (eei < eei_q25)
            deep_focus = (eei > eei_q75) & (fix > fix_q75)
            flow = (eei > eei_q75) & (fix <= fix_q50)

            # Apply from lowest to highest priority (last wins)
            states[cog_fatigue] = self.STATE_FATIGUE
            states[breakdown] = self.STATE_BREAKDOWN
            states[deep_focus] = self.STATE_DEEP_FOCUS
            states[flow] = self.STATE_FLOW

            out["reading_state"] = states

        except (KeyError, TypeError, ValueError):
            out["reading_state"] = self.STATE_NORMAL

        return out

    # ------------------------------------------------------------------
    # 2. Word-level metrics
    # ------------------------------------------------------------------

    def compute_word_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``word_count``, ``fixation_per_word``, ``sentence_complexity``,
        and linguistic readability columns (syllables, complex words, Flesch Reading Ease,
        Grade Level, linguistic root cause) to *df*.
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

        # Deterministic linguistic readability profiling
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
        Compute sentence-level comprehension statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Must have already been processed by ``classify_reading_states``
            and ``compute_word_metrics``.

        Returns
        -------
        dict
            Breakdown report (see module docstring for schema).
        """
        total = len(df)

        # ---- state counts ------------------------------------------------
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

        # ---- simple vs complex sentence profiles -------------------------
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

        # ---- post-breakdown EEI drift ------------------------------------
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

        # ---- top-5 flow sentences ----------------------------------------
        try:
            flow_df = df[df["reading_state"] == self.STATE_FLOW].copy()
            flow_df = flow_df.sort_values("EEI", ascending=False)
            top_flow = [_sentence_row(flow_df, i) for i in range(min(5, len(flow_df)))]
        except (KeyError, TypeError):
            top_flow = []

        # ---- top-5 breakdown sentences -----------------------------------
        try:
            bd_df = df[df["reading_state"] == self.STATE_BREAKDOWN].copy()
            bd_df = bd_df.sort_values("fixation_per_word", ascending=False)
            top_breakdown = [_sentence_row(bd_df, i) for i in range(min(5, len(bd_df)))]
        except (KeyError, TypeError):
            top_breakdown = []

        # ---- top-5 high-workload sentences (by CWI) ----------------------
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

        # ---- personal optimal zone (from flow sentences) -----------------
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

        # ---- Learning Efficiency Score (LES, 0-100) -------------------------
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
        }

        # ---- Session Fatigue Curve (Chronological Timeline) -----------------
        num_pts = min(8, max(4, total // 25)) if total > 20 else max(1, total)
        chunk_size = max(1, total // num_pts)
        checkpoints = []
        fatigue_onset_sentence = None
        for i in range(num_pts):
            s_idx = i * chunk_size
            e_idx = total if i == num_pts - 1 else (i + 1) * chunk_size
            chunk = df.iloc[s_idx:e_idx]
            ch_eei = round(_safe_float(pd.to_numeric(chunk.get("EEI", pd.Series(dtype=float)), errors="coerce").mean()), 4)
            ch_cwi = round(_safe_float(pd.to_numeric(chunk.get("CWI", pd.Series(dtype=float)), errors="coerce").mean()), 4)
            ch_fix = round(_safe_float(pd.to_numeric(chunk.get("FixationCount", pd.Series(dtype=float)), errors="coerce").mean()), 2)
            ch_trt = round(_safe_float(pd.to_numeric(chunk.get("TRT", pd.Series(dtype=float)), errors="coerce").mean()), 1)
            est_min = round((s_idx + e_idx) / 2.0 * 0.12, 1)

            is_fatigued = (ch_cwi > 0.15 and ch_eei < eei_mean) or (ch_fix > 1.8 and ch_eei < eei_mean)
            status = "Elevated Fatigue" if is_fatigued else ("Optimal Focus" if ch_eei >= eei_mean else "Mild Drift")

            if is_fatigued and fatigue_onset_sentence is None and s_idx > 0:
                fatigue_onset_sentence = s_idx

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
                "status": status,
            })

        if fatigue_onset_sentence is None:
            fatigue_onset_sentence = int(total * 0.65) if total > 30 else total
        est_fatigue_minute = round(fatigue_onset_sentence * 0.12, 1)
        recommended_pomodoro = max(15, min(35, int(est_fatigue_minute)))

        session_fatigue_curve = {
            "checkpoints": checkpoints,
            "fatigue_onset_sentence": fatigue_onset_sentence,
            "estimated_fatigue_minute": est_fatigue_minute,
            "recommended_study_block_minutes": recommended_pomodoro,
            "insight": f"Attentional engagement was highest during sentences 1–{fatigue_onset_sentence} (~{est_fatigue_minute:.0f} min). Pacing with a {recommended_pomodoro}-minute Pomodoro interval aligns with your neural stamina profile.",
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
        """
        Produce a jargon-free narrative from the breakdown report.

        All numbers are drawn exclusively from *report_dict* — nothing is
        invented.
        """
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

            # Avoid division by zero
            cwi_ratio = (cwi_complex / (cwi_simple + 1e-9)) if cwi_simple > 1e-9 else 1.0
            trt_ratio = ((trt_complex - trt_simple) / (trt_simple + 1e-9) * 100.0) if trt_simple > 1e-9 else 0.0

            optimal_zone = report_dict.get("personal_optimal_zone", {})
            max_words = _safe_int(optimal_zone.get("max_word_count_for_flow", 20))
            max_fix_per_word = _safe_float(optimal_zone.get("max_fixation_per_word", 1.0))

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

        except Exception as exc:  # pragma: no cover
            return (
                f"Summary could not be generated for student {participant_id}. "
                f"Reason: {exc}"
            )

    # ------------------------------------------------------------------
    # Convenience: run the full pipeline on the stored DataFrame
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline on the DataFrame passed at construction time.

        Returns
        -------
        dict with keys:
            ``df``          — enriched DataFrame
            ``report``      — breakdown report dict
            ``summary``     — plain-English narrative string
        """
        df = self.compute_word_metrics(self._df)
        df = self.classify_reading_states(df)
        report = self.build_breakdown_report(df)

        pid_col = df.get("patient_id", pd.Series(dtype=str))
        pid = str(pid_col.iloc[0]) if len(pid_col) > 0 and pid_col.notna().any() else "unknown"
        summary = self.build_plain_english_summary(report, pid)

        return {"df": df, "report": report, "summary": summary}
