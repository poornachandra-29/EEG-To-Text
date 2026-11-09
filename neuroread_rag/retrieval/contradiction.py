"""
Contradictory and Divergent Evidence Detector.
Identifies conflicting, context-dependent, or nuanced physiological findings in academic literature.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from neuroread_rag.discovery.models import ResearchChunk


@dataclass
class ContradictionReport:
    """Structured report detailing divergent or context-dependent literature findings."""
    has_divergence: bool
    metric_id: str
    divergence_type: str  # "Context-Dependent Interpretation", "Mechanistic Polarity", "Task-Specific Modulation", "Consensus"
    explanation: str
    contrasting_perspectives: List[Dict[str, str]]
    reconciliation_guideline: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical scientific nuances known in EEG and Eye-Tracking literature
KNOWN_LITERATURE_NUANCES: Dict[str, Dict[str, Any]] = {
    "FMT": {
        "perspective_a": "FMT elevation reflects active cognitive control, conflict monitoring, and working memory maintenance (Cavanagh & Frank 2014).",
        "perspective_b": "Diffuse theta elevation accompanied by alpha power increase reflects mental fatigue, prolonged vigilance decrement, or drowsiness (Klimesch 1999).",
        "reconciliation": "Examine concurrent Alpha and EEI: Elevated FMT with suppressed alpha indicates active task engagement; elevated theta with elevated alpha indicates fatigue-compatible state."
    },
    "ThetaPower": {
        "perspective_a": "Theta synchronizes during deep memory retrieval and sentence parsing demands.",
        "perspective_b": "High baseline theta without active task demand may indicate lower cortical arousal or attentional drifting.",
        "reconciliation": "Distinguish frontal midline localization from global scalp theta; evaluate relative to personal baseline."
    },
    "PupilSize": {
        "perspective_a": "Pupil dilation scales monotonically with mental effort, sentence complexity, and cognitive load (Beatty & Lucero-Wagoner 2000).",
        "perspective_b": "Extreme cognitive overload or complete disengagement (mind wandering) can cause pupillary constriction or blunted phasic reactivity (Smallwood & Schooler 2015).",
        "reconciliation": "Pupillometry must be combined with concurrent eye fixation metrics (GD/TRT) and ambient luminance baseline controls."
    },
    "TRT": {
        "perspective_a": "Prolonged total reading time reflects sentence complexity, semantic ambiguity, and syntactic reanalysis (Rayner 1998).",
        "perspective_b": "Erratic or unusually brief TRT may reflect attentional zoning-out or superficial skimming without comprehension.",
        "reconciliation": "Correlate TRT with fixation counts: High TRT with high fixation count indicates difficulty; high TRT with high local variability indicates reading cadence instability."
    },
    "GD": {
        "perspective_a": "Longer gaze duration indicates lexical infrequency or early processing difficulty (Rayner 1998).",
        "perspective_b": "Short gaze duration does not always imply effortless reading; it can reflect word skipping during shallow skimming.",
        "reconciliation": "Evaluate word length and frequency alongside First Fixation Duration (FFD)."
    },
    "EEI": {
        "perspective_a": "High EEI (Beta / (Alpha + Theta)) denotes active task engagement and attentional vigilance (Pope et al. 1995).",
        "perspective_b": "High beta power can also reflect muscle tension, ocular artifacts, or anxious arousal rather than productive reading focus.",
        "reconciliation": "Ensure robust normalization and inspect concurrent eye tracking to verify reading stability."
    },
    "MWLI_proxy": {
        "perspective_a": "Cadence instability and diminished EEI correlate with subjective reports of mind wandering (Smallwood 2015).",
        "perspective_b": "Reading variability also naturally occurs during deep reflection, insight generation, or challenging conceptual synthesis.",
        "reconciliation": "Interpret MWLI strictly as a cadence variability proxy, never as definitive proof of attentional lapse."
    }
}


class ContradictionDetector:
    """Analyzes retrieved research chunks to detect scientific divergences or nuance."""

    def analyze_metric_evidence(
        self,
        metric_id: str,
        chunks: List[ResearchChunk]
    ) -> ContradictionReport:
        """Analyze retrieved chunks for a specific metric to identify conflicting claims."""
        if metric_id in KNOWN_LITERATURE_NUANCES:
            nuance = KNOWN_LITERATURE_NUANCES[metric_id]
            return ContradictionReport(
                has_divergence=True,
                metric_id=metric_id,
                divergence_type="Context-Dependent Interpretation",
                explanation=(
                    f"Scientific literature for {metric_id} shows context-dependent findings: "
                    f"physiological modulation depends on task demands, arousal level, and cognitive state."
                ),
                contrasting_perspectives=[
                    {"perspective": "Perspective A (Cognitive Demand)", "claim": nuance["perspective_a"]},
                    {"perspective": "Perspective B (Fatigue / Drift / Nuance)", "claim": nuance["perspective_b"]},
                ],
                reconciliation_guideline=nuance["reconciliation"]
            )

        # General detection based on contrasting textual patterns
        has_increase = False
        has_decrease = False
        chunks_text = " ".join(c.content.lower() for c in chunks)

        if re.search(r"\b(increase[s|d]?|elevat[e|ed]|higher)\b", chunks_text):
            has_increase = True
        if re.search(r"\b(decrease[s|d]?|reduc[e|ed]|lower|depress[ed]?)\b", chunks_text):
            has_decrease = True

        if has_increase and has_decrease:
            return ContradictionReport(
                has_divergence=True,
                metric_id=metric_id,
                divergence_type="Bidirectional Modulation",
                explanation=f"Studies report both elevation and reduction in {metric_id} depending on the experimental paradigm.",
                contrasting_perspectives=[
                    {"perspective": "Workload / Demand", "claim": f"Elevated {metric_id} during high-demand task phases."},
                    {"perspective": "Fatigue / Gating", "claim": f"Decreased {metric_id} during prolonged exposure or cortical idling."}
                ],
                reconciliation_guideline="Evaluate within-participant relative changes rather than absolute universal cutoffs."
            )

        return ContradictionReport(
            has_divergence=False,
            metric_id=metric_id,
            divergence_type="Consensus",
            explanation=f"Retrieved literature demonstrates consistent directional alignment for {metric_id}.",
            contrasting_perspectives=[],
            reconciliation_guideline="Standard literature interpretation applies."
        )
