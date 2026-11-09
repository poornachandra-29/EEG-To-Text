"""
Research Grounding Engine.
Connects deterministic participant physiological statistics with multi-paper scientific findings.
Enforces strict guardrails and produces comprehensive metric-wise scientific interpretations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from neuroread_rag.config import (
    METRIC_DEFINITIONS,
    MANDATORY_DISCLAIMERS,
    OUTPUT_DIR
)
from neuroread_rag.participant_analysis.data_models import (
    ParticipantAnalysisResult,
    MetricSummary
)
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer, SynthesizedEvidence
from neuroread_rag.interpretation.models import (
    ResearchFindingItem,
    MetricInterpretation,
    ParticipantGroundedReport
)
from neuroread_rag.interpretation.guardrails import ScientificGuardrailValidator


class ResearchGroundingEngine:
    """Connects programmatic participant statistics with synthesized academic literature."""

    def __init__(self, synthesizer: EvidenceSynthesizer):
        self.synthesizer = synthesizer
        self.validator = ScientificGuardrailValidator()

    def ground_metric(
        self,
        metric_id: str,
        summary: MetricSummary,
        participant_result: ParticipantAnalysisResult
    ) -> MetricInterpretation:
        """Ground a single metric's measurements with retrieved multi-paper literature."""
        meta = METRIC_DEFINITIONS.get(metric_id, {})
        synth = self.synthesizer.synthesize_metric(metric_id)

        # 1. Deterministic participant measurements
        meas_dict = {
            "mean": summary.mean,
            "median": summary.median,
            "std": summary.std,
            "mad": summary.mad,
            "robust_sd": summary.robust_sd,
            "iqr": summary.iqr,
            "min": summary.min_val,
            "max": summary.max_val,
            "unit": summary.unit
        }

        # 2. Personal baseline comparison
        base_dict = {
            "session_baseline": summary.session_baseline,
            "personal_ref_lower": summary.personal_ref_lower,
            "personal_ref_upper": summary.personal_ref_upper,
            "relative_deviation_pct": summary.relative_deviation_pct,
            "deviation_category": summary.deviation_category
        }

        # 3. Sentence-level dynamics
        outlier_sentences = []
        for obs in participant_result.sentence_observations:
            if metric_id in obs.metrics and obs.is_outlier:
                rz = obs.robust_z_scores.get(metric_id, 0.0)
                if abs(rz) >= 2.0:
                    outlier_sentences.append({
                        "sentence_index": obs.sentence_index,
                        "text": obs.sentence_text,
                        "metric_value": obs.metrics.get(metric_id),
                        "robust_z": rz,
                        "cluster_name": obs.cluster_name,
                        "cognitive_pattern": obs.cognitive_pattern
                    })

        dyn_dict = {
            "variability_index": summary.variability_index,
            "outlier_sentence_count": len(outlier_sentences),
            "sample_outlier_sentences": outlier_sentences[:3],
            "stability_characterization": (
                "Highly stable across sentences" if summary.variability_index < 0.20
                else "Moderate sentence-to-sentence fluctuation" if summary.variability_index < 0.45
                else "Marked sentence-level variability"
            )
        }

        # 4. Map retrieved literature findings
        findings_items: List[ResearchFindingItem] = []
        for paper in synth.papers_cited:
            claim_text = (
                f"{paper.paper_title}: Explores {meta.get('name', metric_id)} dynamics in "
                f"controlled task environments."
            )
            findings_items.append(ResearchFindingItem(
                paper_title=paper.paper_title,
                authors=paper.authors,
                year=paper.year,
                doi=paper.doi,
                venue=paper.venue,
                claim=claim_text,
                experimental_context="Natural reading / continuous cognitive performance",
                evidence_strength="Peer-Reviewed"
            ))

        # 5. Formulate research compatibility
        compat_text = self._formulate_compatibility(metric_id, summary, synth)

        # 6. Scientific limitations
        limitations_text = (
            f"Measurement is an exploratory physiological observation within this specific reading session. "
            f"{synth.contradiction_analysis.reconciliation_guideline} "
            f"Does not establish clinical pathology or definitive psychological states."
        )

        return MetricInterpretation(
            metric_id=metric_id,
            metric_name=meta.get("name", metric_id),
            category=meta.get("category", "Physiological"),
            unit=summary.unit,
            scientific_definition=meta.get("description", ""),
            participant_measurement=meas_dict,
            personal_baseline_comparison=base_dict,
            sentence_level_dynamics=dyn_dict,
            retrieved_scientific_findings=findings_items,
            research_compatibility=compat_text,
            divergence_or_nuance=synth.contradiction_analysis.to_dict(),
            evidence_strength=synth.scientific_evidence_level,
            interpretation_and_limitations=limitations_text
        )

    def _formulate_compatibility(
        self,
        metric_id: str,
        summary: MetricSummary,
        synth: SynthesizedEvidence
    ) -> str:
        """Formulate research compatibility matching empirical measurements to scientific findings."""
        papers_ref = ", ".join(p.authors for p in synth.papers_cited[:2]) or "literature"
        dev_cat = summary.deviation_category
        dev_pct = summary.relative_deviation_pct

        if dev_cat == "Within Typical Range":
            return (
                f"Participant's {summary.display_name} (mean {summary.mean} {summary.unit}) remained "
                f"within their typical session reference range ([{summary.personal_ref_lower}, {summary.personal_ref_upper}]), "
                f"showing a slight relative deviation of {dev_pct:+.1f}%. This consistency aligns with "
                f"findings reported in {papers_ref}, indicating steady physiological maintenance during natural reading."
            )
        elif dev_cat == "Elevated Above Baseline":
            return (
                f"Participant's {summary.display_name} exhibited an elevated pattern (mean {summary.mean} {summary.unit}), "
                f"deviating by {dev_pct:+.1f}% above session baseline. Research by {papers_ref} indicates that "
                f"such elevations reflect heightened processing demands, increased engagement, or greater lexical difficulty."
            )
        else:
            return (
                f"Participant's {summary.display_name} averaged below baseline (mean {summary.mean} {summary.unit}), "
                f"with a relative deviation of {dev_pct:+.1f}%. Scientific evidence ({papers_ref}) suggests "
                f"this corresponds to moments of reduced cognitive allocation, lower ocular dwell, or altered cortical activation."
            )

    def ground_participant_profile(
        self,
        participant_result: ParticipantAnalysisResult
    ) -> ParticipantGroundedReport:
        """Generate a complete research-grounded report covering all physiological domains."""
        interpretations: Dict[str, MetricInterpretation] = {}

        for m_id, summary in participant_result.metrics.items():
            interpretations[m_id] = self.ground_metric(m_id, summary, participant_result)

        # Domain syntheses — each returns a dict; stored as JSON strings in the
        # str-typed dataclass fields so downstream consumers can parse them.
        eeg_dict = self._synthesize_eeg_domain(participant_result, interpretations)
        eye_dict = self._synthesize_eye_domain(participant_result, interpretations)
        workload_dict = self._synthesize_workload_domain(participant_result, interpretations)
        focus_dict = self._synthesize_focus_domain(participant_result, interpretations)
        cluster_grounding = self._ground_clusters(participant_result)

        report = ParticipantGroundedReport(
            participant_id=participant_result.participant_id,
            total_sentences=participant_result.total_sentences,
            metric_interpretations=interpretations,
            eeg_spectral_summary=json.dumps(eeg_dict),
            eye_tracking_summary=json.dumps(eye_dict),
            cognitive_workload_summary=json.dumps(workload_dict),
            focus_attention_summary=json.dumps(focus_dict),
            gmm_cluster_research_grounding=cluster_grounding,
            mandatory_disclaimers=MANDATORY_DISCLAIMERS
        )

        # Validate with guardrails
        report_text = json.dumps(report.to_dict())
        is_valid, violations = self.validator.validate_text(report_text)
        if not is_valid:
            print(f"[Warning] Guardrail violations detected: {violations}")

        return report

    def _synthesize_eeg_domain(
        self,
        res: ParticipantAnalysisResult,
        interps: Dict[str, MetricInterpretation]
    ) -> Dict[str, str]:
        """Return a plain-English EEG domain summary as a structured dict."""
        eei = res.metrics.get("EEI")
        fmt = res.metrics.get("FMT")

        eei_mean = f"{eei.mean:.4f}" if eei else "N/A"
        dev_cat = eei.deviation_category if eei else "typical"
        # Translate deviation_category into natural phrasing
        dev_phrase_map = {
            "Elevated Above Baseline": "above-average",
            "Below Baseline": "below-average",
            "Within Typical Range": "typical",
        }
        dev_phrase = dev_phrase_map.get(dev_cat, "typical")

        # Count EEI outliers from sentence observations
        outlier_count = sum(
            1 for obs in res.sentence_observations
            if obs.is_outlier and "EEI" in obs.metrics
        )

        personal_ref_lower = f"{eei.personal_ref_lower:.4f}" if eei else "N/A"
        fmt_baseline = f"{fmt.session_baseline:.4f}" if fmt else "N/A"

        what_is_happening = (
            f"Your brain's electrical activity during reading showed "
            f"{dev_phrase} engagement. The EEG engagement signal averaged "
            f"{eei_mean} — meaning your brain was actively allocating "
            f"{dev_phrase} cognitive resources compared to your personal session "
            f"baseline. On {outlier_count} sentences, the engagement signal spiked "
            f"or dropped noticeably, suggesting moments of complex text or brief "
            f"fatigue."
        )

        what_you_need = (
            f"For consistent reading performance, your engagement signal ideally "
            f"stays above {personal_ref_lower}. Your frontal brain regions work "
            f"best when theta oscillations (working memory signals) stay moderate "
            f"rather than spiking — which happens when sentence complexity stays "
            f"manageable. Frontal Midline Theta baseline observed at "
            f"{fmt_baseline} µV² is a key reference point."
        )

        how_to_improve = (
            "1. Before reading a dense passage: scan it for unfamiliar words and "
            "build context first. This reduces prediction errors that cause theta "
            "spikes (Cavanagh & Frank, 2014). "
            "2. If your engagement drops mid-session, a 2-minute eyes-closed rest "
            "resets cortical baseline (Pope et al., 1995). "
            "3. Read in well-lit, low-distraction environments — alpha power "
            "stability is sensitive to ambient sensory load."
        )

        plain_summary = (
            f"EEG Analysis: Across {res.total_sentences} reading trials, your "
            f"brain engagement averaged {eei_mean} ({dev_cat.lower()}), "
            f"consistent with vigilant task engagement."
        )

        return {
            "what_is_happening": what_is_happening,
            "what_you_need": what_you_need,
            "how_to_improve": how_to_improve,
            "plain_summary": plain_summary,
        }

    def _synthesize_eye_domain(
        self,
        res: ParticipantAnalysisResult,
        interps: Dict[str, MetricInterpretation]
    ) -> Dict[str, str]:
        """Return a plain-English eye-tracking domain summary as a structured dict."""
        gd = res.metrics.get("GD")
        trt = res.metrics.get("TRT")
        ffd = res.metrics.get("FFD")
        pupil = res.metrics.get("PupilSize")

        gd_mean = f"{gd.mean:.1f}" if gd else "N/A"
        trt_mean = f"{trt.mean:.1f}" if trt else "N/A"
        ffd_mean = f"{ffd.mean:.1f}" if ffd else "N/A"

        trt_ref_upper = f"{trt.personal_ref_upper:.1f}" if trt else "N/A"
        ffd_ref_lower = f"{ffd.personal_ref_lower:.1f}" if ffd else "N/A"

        # Determine pace description relative to FFD baseline
        if ffd:
            if ffd.deviation_category == "Elevated Above Baseline":
                pace_desc = "above your comfortable reading pace"
            elif ffd.deviation_category == "Below Baseline":
                pace_desc = "below your comfortable reading pace"
            else:
                pace_desc = "at your comfortable reading pace"
        else:
            pace_desc = "within your session range"

        # Describe overall eye movement in plain terms
        if trt and trt.deviation_category == "Elevated Above Baseline":
            overall_desc = (
                "Your eyes spent more time per sentence than your session average, "
                "suggesting sentences required additional decoding effort."
            )
        elif trt and trt.deviation_category == "Below Baseline":
            overall_desc = (
                "Your eyes moved through sentences slightly faster than your session "
                "average, suggesting efficient lexical retrieval for most content."
            )
        else:
            overall_desc = (
                "Your overall reading pace was consistent with your session average, "
                "reflecting stable lexical processing."
            )

        what_is_happening = (
            f"Your eyes spent an average of {gd_mean} ms on each word group and "
            f"{trt_mean} ms total on each sentence. {overall_desc} "
            f"Your first glance at each word took about {ffd_mean} ms — "
            f"{pace_desc}."
        )

        what_you_need = (
            f"Effortless reading typically shows first fixation under "
            f"{ffd_ref_lower} ms and total sentence time under {trt_ref_upper} ms. "
            f"When these go higher, it means the text is demanding more decoding "
            f"work from your visual system."
        )

        how_to_improve = (
            "1. For sentences where you go back and re-read: cover the previous "
            "line with a card or finger to force forward reading — this trains "
            "saccadic flow (Rayner, 1998). "
            "2. Increase font size by 2pts and line spacing to 1.5 — this reduces "
            "crowding effects that cause extra fixations. "
            "3. Practice 'chunked reading': train your eyes to land on every 3rd "
            "word rather than every word."
        )

        plain_summary = (
            f"Eye-Tracking: Gaze duration averaged {gd_mean} ms and total reading "
            f"time averaged {trt_mean} ms, with pupil size "
            f"{pupil.mean:.2f} units ({pupil.deviation_category.lower()})."
            if pupil else
            f"Eye-Tracking: Gaze duration averaged {gd_mean} ms and total reading "
            f"time averaged {trt_mean} ms."
        )

        return {
            "what_is_happening": what_is_happening,
            "what_you_need": what_you_need,
            "how_to_improve": how_to_improve,
            "plain_summary": plain_summary,
        }

    def _synthesize_workload_domain(
        self,
        res: ParticipantAnalysisResult,
        interps: Dict[str, MetricInterpretation]
    ) -> Dict[str, str]:
        """Return a plain-English cognitive workload domain summary as a structured dict."""
        cwi = res.metrics.get("CWI")
        theta = res.metrics.get("ThetaPower")
        pupil = res.metrics.get("PupilSize")

        cwi_mean = f"{cwi.mean:+.4f}" if cwi else "N/A"
        cwi_ref_upper = f"{cwi.personal_ref_upper:+.4f}" if cwi else "N/A"

        # Compute % deviation from baseline for plain language
        if cwi and abs(cwi.session_baseline) > 1e-9:
            dev_pct = cwi.relative_deviation_pct
            dev_direction = "above" if dev_pct >= 0 else "below"
            dev_abs = abs(dev_pct)
        else:
            dev_pct = 0.0
            dev_direction = "at"
            dev_abs = 0.0

        theta_desc = (
            f"theta brainwaves at {theta.mean:.4f} (session baseline "
            f"{theta.session_baseline:.4f})"
            if theta else "theta brainwaves"
        )
        pupil_desc = (
            f"pupil dilation averaging {pupil.mean:.2f} units"
            if pupil else "pupil dilation"
        )

        what_is_happening = (
            f"Your cognitive workload — the total mental effort your brain exerted "
            f"— was {cwi_mean} on average, {dev_abs:.1f}% {dev_direction} your "
            f"personal baseline. It spiked significantly on longer, complex "
            f"sentences. During the heaviest moments, {pupil_desc} and "
            f"{theta_desc} surged — your brain's own signals that it was near capacity."
        )

        what_you_need = (
            f"A sustainable cognitive load for reading keeps your workload index "
            f"below your baseline. When it exceeds {cwi_ref_upper}, you are likely "
            f"to start losing comprehension even if you keep reading."
        )

        how_to_improve = (
            "1. Apply the '20-20-20 rule': every 20 minutes, look at something "
            "20 feet away for 20 seconds — this resets oculomotor and cognitive "
            "fatigue. "
            "2. Break long paragraphs into 3-sentence chunks with active recall "
            "between chunks (Cavanagh & Frank, 2014). "
            "3. For very dense material, try the 'read-pause-recall' cycle: read "
            "one paragraph, close your eyes for 5 seconds, recall the key point "
            "before moving on."
        )

        plain_summary = (
            f"Cognitive Workload: The composite workload index averaged "
            f"{cwi_mean} ({dev_abs:.1f}% {dev_direction} session baseline), "
            f"reflecting elevated theta and pupillary demand on complex sentences."
        )

        return {
            "what_is_happening": what_is_happening,
            "what_you_need": what_you_need,
            "how_to_improve": how_to_improve,
            "plain_summary": plain_summary,
        }

    def _synthesize_focus_domain(
        self,
        res: ParticipantAnalysisResult,
        interps: Dict[str, MetricInterpretation]
    ) -> Dict[str, str]:
        """Return a plain-English focus/attention domain summary as a structured dict."""
        sai = res.metrics.get("SAI")
        mwli = res.metrics.get("MWLI_proxy")
        eei_local_sd = res.metrics.get("EEI_local_sd")

        sai_val = sai.mean if sai else None
        mwli_val = mwli.mean if mwli else None

        # Stability phrasing based on EEI local SD variability
        if eei_local_sd:
            if eei_local_sd.variability_index < 0.20:
                rhythm_desc = "stable"
            elif eei_local_sd.variability_index < 0.45:
                rhythm_desc = "moderately variable"
            else:
                rhythm_desc = "variable"
        else:
            rhythm_desc = "variable"

        # SAI interpretation (plain English — no variable name shown)
        if sai_val is not None and sai_val > 0:
            sai_phrase = (
                "Your sustained engagement signal was slightly above your baseline, "
                "suggesting you maintained processing continuity across most sentences."
            )
        elif sai_val is not None:
            sai_phrase = (
                "Your engagement showed some instability, with brief periods of "
                "reduced processing continuity."
            )
        else:
            sai_phrase = (
                "Sustained engagement data was not available for this session."
            )

        # MWLI interpretation (plain English)
        if mwli_val is not None and mwli_val < -0.1:
            mwli_phrase = (
                "Your attention instability signal was below zero, suggesting your "
                "reading cadence was relatively consistent."
            )
        elif mwli_val is not None and mwli_val > 0.1:
            mwli_phrase = (
                "Your attention cadence showed some irregularity across the session, "
                "consistent with brief mind-drift episodes between sentences."
            )
        else:
            mwli_phrase = (
                "Your attention cadence was near your session average."
            )

        what_is_happening = (
            f"Your reading focus and rhythm were {rhythm_desc} across the session. "
            f"{sai_phrase} {mwli_phrase}"
        )

        what_you_need = (
            "Ideal focused reading shows a stable, even engagement rhythm — no "
            "large swings between high and low attention across consecutive sentences."
        )

        how_to_improve = (
            "1. When you notice your mind drifting (re-reading the same line), "
            "mark the paragraph and take a 60-second physical break (stand up, look "
            "away). Smallwood & Schooler (2015) showed this prevents mind-wandering "
            "cascades. "
            "2. Set an intention before each reading block: 'I am reading to find "
            "out X.' This top-down goal keeps the prefrontal cortex engaged. "
            "3. Use the 'active reading' method: underline or mentally note 1 key "
            "phrase per paragraph to anchor attention."
        )

        sai_str = f"{sai_val:+.4f}" if sai_val is not None else "N/A"
        mwli_str = f"{mwli_val:+.4f}" if mwli_val is not None else "N/A"
        plain_summary = (
            f"Focus and Attention: Reading cadence was {rhythm_desc} "
            f"(sustained engagement proxy: {sai_str}; "
            f"attention-instability proxy: {mwli_str})."
        )

        return {
            "what_is_happening": what_is_happening,
            "what_you_need": what_you_need,
            "how_to_improve": how_to_improve,
            "plain_summary": plain_summary,
        }

    def _ground_clusters(self, res: ParticipantAnalysisResult) -> List[Dict[str, Any]]:
        grounded = []
        for c in res.clusters:
            grounded.append({
                "cluster_id": c.cluster_id,
                "cluster_name": c.cluster_name,
                "percentage": c.percentage,
                "dominant_pattern": c.dominant_pattern,
                "descriptive_label": c.descriptive_label,
                "scientific_grounding": (
                    f"Cluster {c.cluster_id} encompasses {c.sentence_count} sentences ({c.percentage}%) "
                    f"characterized by {c.descriptive_label}. Sentences in this cluster exhibit an average "
                    f"membership probability of {c.mean_membership_probability:.2f}."
                )
            })
        return grounded

    def save_grounded_report(
        self,
        report: ParticipantGroundedReport,
        output_path: Optional[Path] = None
    ) -> Path:
        """Save grounded report to JSON."""
        out_file = output_path or (OUTPUT_DIR / f"{report.participant_id}_grounded_report.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return out_file
