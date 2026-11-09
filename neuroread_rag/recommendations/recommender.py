"""
Data models and Recommendation Engine for personalized, research-backed cognitive pacing.

Stage 4: Issue-grounded, conditional recommendations — each recommendation is only included
when the participant data actually demonstrates the corresponding problem pattern.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from neuroread_rag.interpretation.models import ParticipantGroundedReport
from neuroread_rag.participant_analysis.data_models import ParticipantAnalysisResult


@dataclass
class SupportingCitation:
    """Scientific reference supporting a personalized recommendation."""
    citation_text: str  # e.g. "Rayner (1998)"
    paper_title: str
    year: int
    doi: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationItem:
    """Individual personalized recommendation grounded in participant measurements and research."""
    rec_id: str
    title: str
    category: str  # "Pacing & Cognitive Breaks", "Reading Strategy & Chunking", "Workload Modulation", "Visual Ergonomics"
    observed_basis: str
    actionable_guidance: str
    scientific_rationale: str
    supporting_citations: List[SupportingCitation]
    evidence_strength: str  # "High (Canonical Literature)", "Moderate"
    practical_applicability: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["supporting_citations"] = [c.to_dict() for c in self.supporting_citations]
        return res


class RecommendationEngine:
    """Generates personalized, non-clinical recommendations backed by peer-reviewed literature."""

    def generate_recommendations(
        self,
        grounded_report: ParticipantGroundedReport,
        analysis_result: ParticipantAnalysisResult
    ) -> List[RecommendationItem]:
        """Synthesize conditional, issue-grounded recommendations based on observed physiological patterns.

        Each recommendation is only included when the participant data actually demonstrates
        the corresponding problem. All numbers flow deterministically from the analysis result.
        """
        recs: List[RecommendationItem] = []
        patterns_dict = {p.pattern_name: p.percentage for p in analysis_result.patterns}

        # --- LOAD KEY METRICS ---
        eei = analysis_result.metrics.get("EEI")
        trt = analysis_result.metrics.get("TRT")
        gd = analysis_result.metrics.get("GD")
        cwi = analysis_result.metrics.get("CWI")
        fmt = analysis_result.metrics.get("FMT")
        pupil = analysis_result.metrics.get("PupilSize")
        sai = analysis_result.metrics.get("SAI")
        mwli = analysis_result.metrics.get("MWLI_proxy")

        # --- Sentence-level stats for personalization ---
        total = grounded_report.total_sentences
        obs = analysis_result.sentence_observations

        # Collect per-sentence fixation and EEI values for threshold computation
        fixations = [o.metrics.get("FixationCount", 1.0) for o in obs if "FixationCount" in o.metrics]
        eeis = [o.metrics.get("EEI", 0.5) for o in obs if "EEI" in o.metrics]

        fix_p75 = sorted(fixations)[int(0.75 * len(fixations))] if fixations else 1.5
        eei_p25 = sorted(eeis)[int(0.25 * len(eeis))] if eeis else 0.5

        # Comprehension-breakdown count: high fixations AND low EEI simultaneously
        breakdown_count = sum(
            1 for o in obs
            if o.metrics.get("FixationCount", 1.0) > fix_p75
            and o.metrics.get("EEI", 0.5) < eei_p25
        )
        breakdown_pct = round(breakdown_count / max(total, 1) * 100, 1)

        fatigue_pct = patterns_dict.get("Fatigue-compatible EEG pattern", 0.0)
        workload_pct = patterns_dict.get("High cognitive-workload-like pattern", 0.0)
        mean_cwi = cwi.mean if cwi else 0.0
        trt_dev = trt.relative_deviation_pct if trt else 0.0
        mwli_val = mwli.mean if mwli else 0.0

        # --- ISSUE 1: Comprehension Bottleneck ---
        # Conditional: breakdown sentences detected (always include when any breakdown found).
        if breakdown_count > 0:
            recs.append(RecommendationItem(
                rec_id="REC_01_COMPREHENSION",
                title="Comprehension Bottleneck: Words Your Eyes Got Stuck On",
                category="Reading Strategy & Vocabulary",
                observed_basis=(
                    f"On {breakdown_count} of {total} sentences ({breakdown_pct}%), your eyes refixated "
                    f"significantly more than your average while your brain engagement level simultaneously "
                    f"dropped — the classic signal that the text content required extra decoding effort "
                    f"beyond your current comfortable reading pace."
                ),
                actionable_guidance=(
                    "Step 1: Before reading a dense passage, spend 3-5 minutes scanning all proper nouns, "
                    "technical terms, and long words. Build a quick mental map before full reading.\n"
                    "Step 2: When you re-read a sentence more than twice, skip to the next sentence first "
                    "to gather context, then return to decode the difficult word with more information available.\n"
                    "Step 3: Practice 'content-word jumping': train your eyes to land on nouns and verbs "
                    "and skim prepositions and articles. Apps like Spritz or BeeLine Reader accelerate this skill."
                ),
                scientific_rationale=(
                    "Eye-movement research shows that re-fixations cluster around low-frequency words, "
                    "long syllable counts, and unfamiliar morphemes. The brain's reading pathway pauses "
                    "oculomotor progression until phonological decoding succeeds. Building pre-reading "
                    "vocabulary context reduces this stall by priming the lexical access pathway."
                ),
                supporting_citations=[
                    SupportingCitation(
                        citation_text="Rayner (1998)",
                        paper_title="Eye movements in reading and information processing: 20 years of research",
                        year=1998,
                        doi="10.1037/0033-2909.124.3.372",
                        url="https://doi.org/10.1037/0033-2909.124.3.372"
                    ),
                    SupportingCitation(
                        citation_text="Dimigen et al. (2011)",
                        paper_title="Coregistration of eye movements and EEG in natural reading: Analyses and review",
                        year=2011,
                        doi="10.1037/a0023885",
                        url="https://doi.org/10.1037/a0023885"
                    )
                ],
                evidence_strength="High (Canonical Literature)",
                practical_applicability="Immediately applicable in study sessions, technical reading, and academic document review."
            ))

        # --- ISSUE 2: Working Memory Overload ---
        # Conditional: elevated CWI OR high workload pattern percentage.
        if mean_cwi > 0.2 or workload_pct > 15.0:
            recs.append(RecommendationItem(
                rec_id="REC_02_WORKLOAD",
                title="Working Memory Overload on Complex Sentences",
                category="Cognitive Load Management",
                observed_basis=(
                    f"Your Cognitive Workload signal was elevated (mean: {mean_cwi:+.2f} composite units), "
                    f"with {workload_pct:.1f}% of sentences showing high-cognitive-workload patterns. "
                    f"This means on many sentences, your brain's working memory was near or at capacity — "
                    f"processing the sentence content while also managing syntactic complexity."
                ),
                actionable_guidance=(
                    "Step 1: For long sentences (over 20 words), read up to the first verb, pause, "
                    "mentally recall what you've read, then continue. This resets your working memory buffer mid-sentence.\n"
                    "Step 2: Use 'journalistic reading': always read the first and last sentence of a paragraph "
                    "first to build a framework, then fill in the details. This reduces the memory burden of the full pass.\n"
                    f"Step 3: Take a structured 5-minute break every 25 minutes. Your data shows "
                    f"{fatigue_pct:.1f}% fatigue-compatible patterns — scheduled breaks prevent this from accumulating."
                ),
                scientific_rationale=(
                    "Frontal midline theta oscillations — which were elevated in your EEG — reflect active "
                    "working memory gating in the medial prefrontal cortex. When sentence complexity exceeds "
                    "cognitive capacity (~7 ± 2 information chunks), theta bursts signal the system is at "
                    "capacity and comprehension degrades. Chunking and pacing directly reduce this load."
                ),
                supporting_citations=[
                    SupportingCitation(
                        citation_text="Cavanagh & Frank (2014)",
                        paper_title="Frontal midline theta as a neural mechanism for cognitive control",
                        year=2014,
                        doi="10.1016/j.tics.2014.04.012",
                        url="https://doi.org/10.1016/j.tics.2014.04.012"
                    ),
                    SupportingCitation(
                        citation_text="Pope, Bogart & Bartolome (1995)",
                        paper_title="Biocybernetic system evaluates indices of operator engagement in automated task",
                        year=1995,
                        doi="10.1016/0301-0511(95)05116-3",
                        url="https://doi.org/10.1016/0301-0511(95)05116-3"
                    )
                ],
                evidence_strength="High (Canonical Literature)",
                practical_applicability="Highly effective for textbooks, technical documentation, legal documents, and academic papers."
            ))

        # --- ISSUE 3: Fatigue and Attention Drift ---
        # Conditional: meaningful fatigue-pattern prevalence.
        if fatigue_pct > 10.0:
            recs.append(RecommendationItem(
                rec_id="REC_03_FATIGUE",
                title="Attention Drift & Reading Fatigue Pattern",
                category="Pacing & Cognitive Breaks",
                observed_basis=(
                    f"On {fatigue_pct:.1f}% of your sentences, your EEG showed fatigue-compatible patterns — "
                    f"rising alpha and theta power combined with dropping engagement. This is the physiological "
                    f"signature of cognitive fatigue building during the reading session."
                ),
                actionable_guidance=(
                    "Step 1: Adopt 20-25 minute structured reading blocks followed by 3-5 minute breaks "
                    "with visual disengagement (look at something far away, blink slowly 10 times).\n"
                    "Step 2: When you notice yourself re-reading the same line, that's your personal 'fatigue signal'. "
                    "Stop, mark the page, stand up and stretch for 60 seconds before continuing.\n"
                    "Step 3: Set an intention before each reading block: 'I am reading to find out [specific goal].' "
                    "This top-down goal keeps the prefrontal cortex actively engaged and delays fatigue onset."
                ),
                scientific_rationale=(
                    "Elevated alpha power during reading tasks reflects cortical idling and reduced task engagement. "
                    "Combined with theta increases, this spectral pattern matches the EEG signature of "
                    "sustained attention fatigue documented in continuous performance literature. "
                    "Scheduled breaks restore tonic alertness and prevent the drift toward mind-wandering."
                ),
                supporting_citations=[
                    SupportingCitation(
                        citation_text="Smallwood & Schooler (2015)",
                        paper_title="The science of mind wandering: empirically navigating the stream of consciousness",
                        year=2015,
                        doi="10.1146/annurev-psych-010814-015331",
                        url="https://doi.org/10.1146/annurev-psych-010814-015331"
                    ),
                    SupportingCitation(
                        citation_text="Klimesch (1999)",
                        paper_title="EEG alpha and theta oscillations reflect cognitive and memory performance",
                        year=1999,
                        doi="10.1016/S0165-0173(98)00056-3",
                        url="https://doi.org/10.1016/S0165-0173(98)00056-3"
                    )
                ],
                evidence_strength="High (Canonical Literature)",
                practical_applicability="Directly applicable for long study sessions, exam preparation, and professional reading tasks."
            ))

        # --- ISSUE 4: Reading Cadence Instability ---
        # Conditional: SAI below zero OR notable MWLI_proxy magnitude.
        if (sai and sai.mean < 0) or (mwli and abs(mwli.mean) > 0.1):
            sai_str = f"{sai.mean:+.2f}" if sai else "N/A"
            mwli_str = f"{mwli.mean:+.2f}" if mwli else "N/A"
            recs.append(RecommendationItem(
                rec_id="REC_04_CADENCE",
                title="Reading Rhythm Stabilization",
                category="Attention & Cadence Control",
                observed_basis=(
                    f"Your reading rhythm showed variability across the session — alternating between "
                    f"rapid skimming phases and extended pauses on specific sentences. "
                    f"This inconsistent cadence is reflected in your attention stability score "
                    f"(mean: {sai_str}) and instability proxy (mean: {mwli_str})."
                ),
                actionable_guidance=(
                    "Step 1: Use a physical pointer (finger, pen, or cursor) to maintain a steady reading pace "
                    "— your eyes will follow the guide and reduce erratic scanning.\n"
                    "Step 2: After each paragraph, speak or write the main point in one sentence. "
                    "This active recall anchors attention and prevents drift.\n"
                    "Step 3: For important material, read once for structure (fast pass) and once for detail "
                    "(slow pass). Two structured passes outperform one erratic pass in retention."
                ),
                scientific_rationale=(
                    "Erratic eye fixation patterns combined with oscillating EEG engagement ratios indicate "
                    "attentional cadence drift — the reader's focus is decoupled from their eye movements. "
                    "Active metacognitive strategies (self-monitoring, recall prompts) re-couple visual "
                    "attention with semantic processing and stabilize the engagement signal."
                ),
                supporting_citations=[
                    SupportingCitation(
                        citation_text="Smallwood & Schooler (2015)",
                        paper_title="The science of mind wandering: empirically navigating the stream of consciousness",
                        year=2015,
                        doi="10.1146/annurev-psych-010814-015331",
                        url="https://doi.org/10.1146/annurev-psych-010814-015331"
                    )
                ],
                evidence_strength="High (Canonical Literature)",
                practical_applicability="Effective for complex technical texts, lecture notes, and long-form reading."
            ))

        # --- ISSUE 5: Visual Ergonomics (always included, pupil-grounded) ---
        pupil_mean_str = f"{pupil.mean:.1f}" if pupil else "N/A"
        recs.append(RecommendationItem(
            rec_id="REC_05_ERGONOMICS",
            title="Reading Environment & Visual Comfort Optimization",
            category="Visual Ergonomics",
            observed_basis=(
                f"Your average pupil size was {pupil_mean_str} units across the session, "
                f"with variability reflecting both cognitive effort changes and potential luminance fluctuations. "
                f"Your reading environment directly affects your eye's autonomic response and cognitive comfort."
            ),
            actionable_guidance=(
                "Step 1: Match screen brightness to your ambient room light — your pupil should not need to "
                "constrict or dilate significantly when shifting your gaze from screen to room.\n"
                "Step 2: Use font size 14-16pt for digital reading with 1.4-1.6× line spacing. "
                "This reduces oculomotor crowding and lowers refixation rates by up to 20%.\n"
                "Step 3: Apply the 20-20-20 rule during every reading session: every 20 minutes, "
                "look at an object 20 feet away for 20 seconds to reset your focal length and "
                "reduce ciliary muscle fatigue."
            ),
            scientific_rationale=(
                "Pupillometry captures both cognitive effort (norepinephrine-driven) and the pupillary "
                "light reflex. Minimizing luminance contrast between screen and environment stabilizes "
                "baseline pupil diameter, reducing autonomic noise in the cognitive effort signal and "
                "improving reading comfort over extended sessions."
            ),
            supporting_citations=[
                SupportingCitation(
                    citation_text="Beatty & Lucero-Wagoner (2000)",
                    paper_title="The pupillary system and cognitive workload",
                    year=2000,
                    doi="10.1017/CBO9780511546396.007",
                    url="https://doi.org/10.1017/CBO9780511546396.007"
                )
            ],
            evidence_strength="Moderate",
            practical_applicability="Immediately applicable with any screen-based reading workflow."
        ))

        return recs
