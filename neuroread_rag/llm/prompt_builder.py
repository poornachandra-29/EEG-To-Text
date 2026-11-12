"""
Prompt Builder — transforms a student's precomputed CSV into a structured
context prompt for the LLM, and injects retrieved scientific literature.

This is the intellectual core of the LLM integration: the quality of the
personalized recommendations depends entirely on how we structure the context.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================================== #
#  SYSTEM PROMPT (Persona + Hard Constraints)
# =========================================================================== #
SYSTEM_PROMPT = """\
You are NeuroRead AI, an evidence-based cognitive learning advisor for students.

You analyze students' EEG brainwave patterns and eye-tracking data recorded
during natural reading to provide personalized, scientifically-grounded
study recommendations that help them learn more effectively.

STRICT RULES YOU MUST FOLLOW:
1. You are NOT a doctor. NEVER diagnose ADHD, dyslexia, or any clinical condition.
   NEVER use words like "disorder", "abnormal", "pathology", or "diagnosis".
2. Every recommendation MUST reference the student's specific data values
   (e.g., "your theta/beta ratio of 1.42" not "your elevated theta/beta ratio").
3. Every scientific claim MUST cite a paper from the provided research context.
   Do NOT invent citations or claim facts not in the provided literature.
4. Use encouraging, growth-mindset language. Frame everything as optimization,
   not deficiency. The student is not "broken" — their brain data gives us
   signals about how to optimize their study approach.
5. Provide exactly 3-5 actionable recommendations, each with:
   - What the data shows (cite specific numbers from the profile)
   - Why it matters (cite the scientific paper from provided context)
   - A concrete, personalized 3-step action plan with specific timings
     calculated from the student's own data (not generic advice)
6. All timings must be calculated from the student's fatigue onset data.
   For example: if fatigue onset is at sentence 180 of 250, and their average
   reading speed is ~7 sentences/minute, their optimal study block is ~25 minutes.
7. Use Markdown formatting with headers, bullet points, and bold text.
8. Keep the total response under 1500 words.

OUTPUT STRUCTURE (follow exactly):
## 🧠 Reading Personality: [2-3 word label]
[1-2 sentence personality summary based on their dominant cognitive pattern]

## ✅ Your Strengths
[3 bullet points identifying specific strengths from the data]

## 📊 Personalized Recommendations
### Recommendation 1: [Title]
**What your data shows:** [cite specific numbers]
**Why it matters:** [cite research paper]
**Your 3-step action plan:**
1. [step with personalized timing]
2. [step with specific strategy]
3. [step with measurable goal]

[Repeat for 2-4 more recommendations]

## 📅 Your Personalized Study Schedule
[Weekly schedule suggestion based on their fatigue pattern and optimal session length]

## 💪 Keep Going!
[Motivational closing referencing their specific strengths]
"""

# =========================================================================== #
#  Curated Scientific Literature for RAG Context
# =========================================================================== #
RAG_LITERATURE_CONTEXT = """\
### Scientific Literature (use these for citations — do NOT invent others)

[1] Rayner, K. (1998). "Eye movements in reading and information processing:
    20 years of research." Psychological Bulletin, 124(3), 372-422.
    KEY FINDINGS: First fixation durations (FFD) of 200-250ms are typical for
    skilled readers. Gaze duration (GD) reflects early lexical access difficulty.
    Total reading time (TRT) includes regressions and syntactic reanalysis.
    Fixation counts > 1.5 per word indicate decoding difficulty.
    DOI: 10.1037/0033-2909.124.3.372

[2] Cavanagh, J. F., & Frank, M. J. (2014). "Frontal midline theta as a
    neural mechanism for cognitive control." Trends in Cognitive Sciences, 18(8).
    KEY FINDINGS: Frontal midline theta (FMT, 4-8 Hz) increases with working
    memory load, task difficulty, and conflict monitoring. Elevated FMT signals
    the prefrontal cortex is actively gating information.
    DOI: 10.1016/j.tics.2014.04.012

[3] Klimesch, W. (1999). "EEG alpha and theta oscillations reflect cognitive
    and memory performance." Brain Research Reviews, 29(2-3), 169-195.
    KEY FINDINGS: Alpha suppression (desynchronization) reflects active cortical
    processing. High alpha = cortical idling/disengagement. Theta increases
    during encoding and working memory maintenance. Theta/Alpha ratio > 1.0
    suggests heavy working memory utilization.
    DOI: 10.1016/S0165-0173(98)00056-3

[4] Pope, Bogart & Bartolome (1995). "Biocybernetic system evaluates indices
    of operator engagement." Biological Psychology, 40, 187-195.
    KEY FINDINGS: EEG Engagement Index (EEI = Beta / (Alpha + Theta)) tracks
    active attentive processing. Higher EEI = more engaged. Lower EEI = fatigue,
    disengagement, or mind-wandering.
    DOI: 10.1016/0301-0511(95)05116-3

[5] Beatty & Lucero-Wagoner (2000). "The pupillary system and cognitive workload."
    KEY FINDINGS: Pupil diameter increases with memory load and mental effort.
    Baseline dilation reflects tonic vigilance. Phasic dilations mark
    instantaneous cognitive effort spikes.
    DOI: 10.1017/CBO9780511546396.007

[6] Smallwood & Schooler (2015). "The science of mind wandering." Annual Review
    of Psychology, 66, 487-518.
    KEY FINDINGS: Mind wandering involves perceptual decoupling from text.
    Markers: erratic gaze, increased reading time variability, diminished
    pupillary reactivity, reduced EEG engagement. These capture attention
    instability without establishing clinical pathology.
    DOI: 10.1146/annurev-psych-010814-015331

[7] Roediger & Butler (2011). "The critical role of retrieval practice in
    long-term retention." Trends in Cognitive Sciences, 15(1), 20-27.
    KEY FINDINGS: Active recall (testing yourself) strengthens memory traces
    far more than passive re-reading. Spaced retrieval practice produces
    durable long-term learning.
    DOI: 10.1016/j.tics.2010.09.003

[8] Sweller, J. (1988). "Cognitive load during problem solving: Effects on
    learning." Cognitive Science, 12(2), 257-285.
    KEY FINDINGS: Working memory capacity is limited (~4±1 chunks). When
    cognitive load exceeds capacity, encoding into long-term memory fails.
    Chunking, scaffolding, and pacing reduce extraneous load.
    DOI: 10.1207/s15516709cog1202_4

[9] Cepeda et al. (2006). "Distributed practice in verbal recall tasks."
    Review of Educational Research, 76(3), 354-380.
    KEY FINDINGS: Spaced repetition (distributed practice) produces significantly
    better long-term retention than massed practice (cramming). Optimal spacing
    intervals increase with desired retention duration.

[10] Dimigen et al. (2011). "Coregistration of eye movements and EEG in
     natural reading." J. Experimental Psychology: General, 140(4), 552.
     KEY FINDINGS: Fixation-related potentials link eye-tracking metrics to
     neural processing. High fixation counts + elevated theta = active
     semantic processing difficulty.
     DOI: 10.1037/a0023885
"""


# =========================================================================== #
#  User Prompt Template
# =========================================================================== #
USER_PROMPT_TEMPLATE = """\
## Student Cognitive Profile: {participant_id}
**Reading Task:** {task_name} | **Sentences Analyzed:** {total_sentences}

---

### EEG Brain Engagement Metrics
| Metric | Value | Session Baseline | Deviation |
|:---|:---:|:---:|:---:|
| EEG Engagement Index (EEI) | {eei_mean:.4f} | {eei_baseline:.4f} | {eei_dev:+.1f}% |
| Frontal Midline Theta (FMT) | {fmt_mean:.4f} | — | working memory proxy |
| Alpha Suppression Mag. (ASM) | {asm_mean:.4f} | — | attention depth |
| Theta / Beta Ratio | {tbr:.4f} | — | attentional control (lower = better) |
| Theta / Alpha Ratio | {tar:.4f} | — | memory load indicator |

### Eye-Tracking Reading Behavior
| Metric | Value | Interpretation |
|:---|:---:|:---|
| Gaze Duration (GD) | {gd_median:.1f} ms | first-pass reading time per word |
| First Fixation Duration (FFD) | {ffd_median:.1f} ms | initial word recognition speed |
| Total Reading Time (TRT) | {trt_median:.1f} ms | including re-reads |
| Fixation Count | {fix_mean:.2f} per word | >1.5 = re-reading difficulty |
| Pupil Size | {pupil_mean:.1f} units | cognitive effort / luminance |

### Cognitive State Clustering (GMM, K=6)
- **Primary Cluster:** {dominant_cluster} ({dominant_cluster_pct:.1f}% of sentences)
- **Secondary Cluster:** {second_cluster} ({second_cluster_pct:.1f}% of sentences)

### Cognitive Pattern Distribution
{pattern_distribution}

### Session Dynamics & Fatigue Analysis
- **Flow-state sentences** (smooth, efficient reading): {flow_pct:.1f}%
- **Comprehension breakdown sentences** (high fixations + low engagement): {breakdown_pct:.1f}%
- **Fatigue-compatible sentences** (rising alpha, dropping engagement): {fatigue_pct:.1f}%
- **High cognitive workload sentences:** {workload_pct:.1f}%
- **Estimated fatigue onset:** sentence ~{fatigue_onset} of {total_sentences} (~{fatigue_minutes:.0f} min into session)
- **Estimated optimal study block:** ~{optimal_block_min:.0f} minutes before a break

{rag_literature}

---

Based on this student's specific cognitive profile, generate a fully personalized
learning assessment and study optimization report. Remember: cite specific numbers
from the profile above, calculate timings from THEIR fatigue data, and cite papers
from the provided literature ONLY.
"""


class CognitivePromptBuilder:
    """Transforms precomputed student CSVs into structured LLM prompts with dynamic RAG retrieval."""

    def __init__(self, enable_dynamic_retrieval: bool = True):
        self.enable_dynamic_retrieval = enable_dynamic_retrieval
        self._retriever = None
        if enable_dynamic_retrieval:
            try:
                from neuroread_rag.ingestion.vector_store import HybridVectorStore
                from neuroread_rag.retrieval.hybrid_search import HybridRetriever
                vs = HybridVectorStore()
                self._retriever = HybridRetriever(vs)
                logger.info("CognitivePromptBuilder: Dynamic Hybrid RAG Retriever initialized.")
            except Exception as e:
                logger.warning("Could not initialize dynamic vector store: %s", e)

    def retrieve_dynamic_rag_context(self, profile: Dict[str, Any], max_chunks: int = 5) -> str:
        """
        True RAG Engine:
        Formulate targeted queries from the student's physiological anomalies,
        dynamically retrieve matching peer-reviewed research chunks from the
        vector store, and format them for context augmentation.
        """
        if not self._retriever:
            return RAG_LITERATURE_CONTEXT

        # 1. Formulate student-specific scientific queries based on physiological indicators
        targeted_queries = []

        # Indicator A: Working Memory & Cognitive Workload
        fmt_val = profile.get("fmt_mean", 0.0)
        workload_pct = profile.get("workload_pct", 0.0)
        if fmt_val > 0.25 or workload_pct > 10.0:
            targeted_queries.append((
                "Frontal midline theta working memory cognitive workload control",
                f"Student FMT is {fmt_val:.4f} with {workload_pct:.1f}% high workload sentences"
            ))

        # Indicator B: Decoding, Lexical Access & Fixation Count
        fix_val = profile.get("fix_mean", 1.0)
        breakdown_pct = profile.get("breakdown_pct", 0.0)
        if fix_val > 1.4 or breakdown_pct > 5.0:
            targeted_queries.append((
                "Eye movements gaze duration fixation count lexical decoding reading comprehension",
                f"Student FixationCount is {fix_val:.2f}/word with {breakdown_pct:.1f}% comprehension breakdown"
            ))

        # Indicator C: Attentional Engagement, Fatigue & Mind Wandering
        fatigue_pct = profile.get("fatigue_pct", 0.0)
        eei_dev = profile.get("eei_dev", 0.0)
        if fatigue_pct > 8.0 or eei_dev < -5.0:
            targeted_queries.append((
                "EEG engagement index alpha theta spectral fatigue vigilance mind wandering",
                f"Student fatigue pattern prevalence is {fatigue_pct:.1f}% with EEI deviation {eei_dev:+.1f}%"
            ))

        # Indicator D: Pupillary Reactivity & Autonomic Effort
        pupil_val = profile.get("pupil_mean", 0.0)
        if pupil_val > 500.0:
            targeted_queries.append((
                "Pupillometry pupil diameter cognitive effort autonomic arousal memory load",
                f"Student pupil size is {pupil_val:.1f} units"
            ))

        # Fallback foundational query
        if not targeted_queries:
            targeted_queries.append((
                "Simultaneous EEG eye tracking natural reading comprehension benchmark",
                "Baseline reading comprehension profile"
            ))

        # 2. Query Hybrid Vector Store and deduplicate
        retrieved_docs = []
        seen_chunk_ids = set()

        for query_str, match_reason in targeted_queries:
            try:
                candidates = self._retriever.retrieve(query_str, top_k=2)
                for chunk, score in candidates:
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        retrieved_docs.append((chunk, score, match_reason))
            except Exception as e:
                logger.warning("RAG retrieval error for query '%s': %s", query_str, e)

        if not retrieved_docs:
            return RAG_LITERATURE_CONTEXT

        # 3. Format retrieved documents for LLM context injection
        doc_blocks = ["### Dynamically Retrieved Research Evidence (from NeuroRead Vector Store)\n"]
        for i, (chunk, score, reason) in enumerate(retrieved_docs[:max_chunks], 1):
            doi_str = f"https://doi.org/{chunk.doi}" if chunk.doi else "N/A"
            doc_blocks.append(
                f"[Evidence Document {i}] \"{chunk.paper_title}\"\n"
                f"- **Authors & Year:** {chunk.authors_short} ({chunk.year})\n"
                f"- **DOI:** {doi_str}\n"
                f"- **Targeted Student Metric:** {reason}\n"
                f"- **Scientific Findings & Guidance:** {chunk.content.strip()}\n"
            )

        return "\n".join(doc_blocks)

    def build_from_csv(self, csv_path: Path, rag_context: Optional[str] = None) -> str:
        """Read a precomputed student CSV and build a complete user prompt with dynamic RAG context."""
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV is empty: {csv_path}")

        profile = self._extract_profile(df)
        # Use dynamic RAG context if not provided
        profile["rag_literature"] = rag_context or self.retrieve_dynamic_rag_context(profile)
        return USER_PROMPT_TEMPLATE.format(**profile)

    def build_from_dataframe(
        self, df: pd.DataFrame, rag_context: Optional[str] = None
    ) -> str:
        """Build a prompt directly from an in-memory DataFrame with dynamic RAG context."""
        profile = self._extract_profile(df)
        profile["rag_literature"] = rag_context or self.retrieve_dynamic_rag_context(profile)
        return USER_PROMPT_TEMPLATE.format(**profile)


    def get_system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------ #
    #  Profile Extraction from CSV DataFrame
    # ------------------------------------------------------------------ #
    def _extract_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract all profile metrics from the student's precomputed CSV."""

        pid = df["patient_id"].iloc[0] if "patient_id" in df.columns else "Unknown"
        task = df["task_name"].iloc[0] if "task_name" in df.columns else "Unknown"
        total = len(df)

        # --- Core EEG Metrics ---
        eei_mean = df["EEI"].mean() if "EEI" in df.columns else 0.0
        fmt_mean = df["FMT"].mean() if "FMT" in df.columns else 0.0
        asm_mean = df["ASM"].mean() if "ASM" in df.columns else 0.0

        # Compute EEI session baseline (median) and deviation
        eei_median = df["EEI"].median() if "EEI" in df.columns else eei_mean
        eei_dev = ((eei_mean - eei_median) / max(abs(eei_median), 1e-8)) * 100

        # Theta/Beta and Theta/Alpha ratios
        tbr = df["Theta_Beta_Ratio"].mean() if "Theta_Beta_Ratio" in df.columns else 0.0
        tar = df["Theta_Alpha_Ratio"].mean() if "Theta_Alpha_Ratio" in df.columns else 0.0

        # --- Eye-Tracking Metrics ---
        gd_median = df["GD"].median() if "GD" in df.columns else 0.0
        ffd_median = df["FFD"].median() if "FFD" in df.columns else 0.0
        trt_median = df["TRT"].median() if "TRT" in df.columns else 0.0
        fix_mean = df["FixationCount"].mean() if "FixationCount" in df.columns else 0.0
        pupil_mean = df["PupilSize"].mean() if "PupilSize" in df.columns else 0.0

        # --- Cluster Distribution ---
        cluster_col = "cluster_name" if "cluster_name" in df.columns else "cluster_id"
        cluster_counts = df[cluster_col].value_counts(normalize=True) * 100
        dominant_cluster = cluster_counts.index[0] if len(cluster_counts) > 0 else "Unknown"
        dominant_cluster_pct = cluster_counts.iloc[0] if len(cluster_counts) > 0 else 0.0
        second_cluster = cluster_counts.index[1] if len(cluster_counts) > 1 else "N/A"
        second_cluster_pct = cluster_counts.iloc[1] if len(cluster_counts) > 1 else 0.0

        # --- Cognitive Pattern Distribution ---
        pattern_col = "cognitive_pattern"
        if pattern_col in df.columns:
            pattern_counts = df[pattern_col].value_counts()
            pattern_lines = []
            for pat, cnt in pattern_counts.items():
                pct = cnt / total * 100
                pattern_lines.append(f"- **{pat}**: {cnt} sentences ({pct:.1f}%)")
            pattern_distribution = "\n".join(pattern_lines)

            fatigue_pct = (
                pattern_counts.get("Fatigue-compatible EEG pattern", 0) / total * 100
            )
            workload_pct = (
                pattern_counts.get("High cognitive-workload-like pattern", 0) / total * 100
            )
        else:
            pattern_distribution = "- Pattern data not available"
            fatigue_pct = 0.0
            workload_pct = 0.0

        # --- Flow / Breakdown Detection ---
        flow_pct, breakdown_pct = self._compute_flow_breakdown(df)

        # --- Fatigue Onset Estimation ---
        fatigue_onset, fatigue_minutes, optimal_block = self._estimate_fatigue_onset(
            df, total
        )

        return {
            "participant_id": pid,
            "task_name": task,
            "total_sentences": total,
            "eei_mean": eei_mean,
            "eei_baseline": eei_median,
            "eei_dev": eei_dev,
            "fmt_mean": fmt_mean,
            "asm_mean": asm_mean,
            "tbr": tbr,
            "tar": tar,
            "gd_median": gd_median,
            "ffd_median": ffd_median,
            "trt_median": trt_median,
            "fix_mean": fix_mean,
            "pupil_mean": pupil_mean,
            "dominant_cluster": dominant_cluster,
            "dominant_cluster_pct": dominant_cluster_pct,
            "second_cluster": second_cluster,
            "second_cluster_pct": second_cluster_pct,
            "pattern_distribution": pattern_distribution,
            "flow_pct": flow_pct,
            "breakdown_pct": breakdown_pct,
            "fatigue_pct": fatigue_pct,
            "workload_pct": workload_pct,
            "fatigue_onset": fatigue_onset,
            "fatigue_minutes": fatigue_minutes,
            "optimal_block_min": optimal_block,
        }

    # ------------------------------------------------------------------ #
    #  Flow / Breakdown Analysis
    # ------------------------------------------------------------------ #
    def _compute_flow_breakdown(self, df: pd.DataFrame) -> tuple:
        """Estimate flow-state and comprehension-breakdown sentence percentages."""
        total = len(df)
        if total == 0:
            return 0.0, 0.0

        if "EEI" in df.columns and "FixationCount" in df.columns:
            eei_vals = df["EEI"].dropna()
            fix_vals = df["FixationCount"].dropna()

            if len(eei_vals) > 0 and len(fix_vals) > 0:
                eei_q75 = eei_vals.quantile(0.75)
                eei_q25 = eei_vals.quantile(0.25)
                fix_q75 = fix_vals.quantile(0.75)
                fix_q25 = fix_vals.quantile(0.25)

                # Flow: high engagement + low fixations (smooth reading)
                flow_mask = (df["EEI"] >= eei_q75) & (df["FixationCount"] <= fix_q25)
                flow_pct = flow_mask.sum() / total * 100

                # Breakdown: low engagement + high fixations (struggling)
                breakdown_mask = (df["EEI"] <= eei_q25) & (df["FixationCount"] >= fix_q75)
                breakdown_pct = breakdown_mask.sum() / total * 100

                return round(flow_pct, 1), round(breakdown_pct, 1)

        return 0.0, 0.0

    # ------------------------------------------------------------------ #
    #  Fatigue Onset Estimation
    # ------------------------------------------------------------------ #
    def _estimate_fatigue_onset(
        self, df: pd.DataFrame, total: int
    ) -> tuple:
        """Estimate where in the session fatigue patterns begin to dominate."""

        # Default: assume fatigue starts at ~75% of the session
        default_onset = int(total * 0.75)
        avg_reading_speed = 7.0  # approximate sentences per minute

        if "cognitive_pattern" in df.columns and total > 20:
            # Find the first sustained window where fatigue dominates
            is_fatigue = (
                df["cognitive_pattern"] == "Fatigue-compatible EEG pattern"
            ).astype(int)

            # Use a rolling window of 10 sentences
            window = min(10, total // 5)
            if window >= 3:
                rolling_fatigue = is_fatigue.rolling(window=window, min_periods=1).mean()
                # Fatigue onset = first point where >50% of the window is fatigued
                fatigue_indices = rolling_fatigue[rolling_fatigue > 0.5].index
                if len(fatigue_indices) > 0:
                    onset = int(fatigue_indices[0])
                    minutes = onset / avg_reading_speed
                    optimal = max(10, minutes * 0.85)  # 85% of fatigue onset time
                    return onset, minutes, optimal

        # Fallback: use EEI declining trend
        if "EEI" in df.columns and total > 20:
            eei = df["EEI"].values
            # Split into thirds
            third = total // 3
            first_third_mean = np.nanmean(eei[:third])
            last_third_mean = np.nanmean(eei[2 * third:])
            if first_third_mean > 0 and last_third_mean < first_third_mean * 0.9:
                # EEI dropped >10% from first to last third → fatigue onset ~2/3
                onset = 2 * third
                minutes = onset / avg_reading_speed
                optimal = max(10, minutes * 0.85)
                return onset, minutes, optimal

        # Default fallback
        minutes = default_onset / avg_reading_speed
        optimal = max(10, minutes * 0.85)
        return default_onset, minutes, optimal
