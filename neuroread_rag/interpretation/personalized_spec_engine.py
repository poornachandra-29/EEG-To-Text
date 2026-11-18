"""
Personalized Metric Specification & Pedagogical Diagnosis Engine.
Pure RAG architecture: grounds retrieved peer-reviewed scientific papers with student CSV data
to generate tailored, student-centric specifications for every physiological metric:
1. What this metric says (Scientific literature ground truth)
2. How you are performing according to this metric (Personalized baseline comparison & CSV review)
3. How to improve the results based on this metric in the context of this metric (Actionable study strategy)
4. What improving this metric actually means to them (Cognitive & pedagogical impact)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from neuroread_rag.participant_analysis.sentence_comparator import analyze_sentence_linguistics
from neuroread_rag.interpretation.llm_metric_rag import LLMMetricRAGGenerator
from neuroread_rag.retrieval.synthesizer import SynthesizedEvidence
from neuroread_rag.ingestion.vector_store import HybridVectorStore


METRIC_FUNCTIONAL_ROLES = {
    "EEI": "Tracks your real-time mental alertness and active cortical engagement across sentences.",
    "FMT": "Measures executive cognitive control and working memory load during complex semantic integration.",
    "theta_fm": "Measures prefrontal midline theta power, tracking active working memory allocation.",
    "TRT": "Measures cumulative ocular dwell time on a sentence, reflecting both initial reading and syntactic rereading.",
    "FixationCount": "Tracks the total number of eye pauses, reflecting lexical processing effort and ocular regressions.",
    "GD": "Captures first-pass gaze duration before saccading away, reflecting initial word recognition difficulty.",
    "FFD": "Measures First Fixation Duration during initial visual contact with a word.",
    "PupilSize": "Monitors pupillary dilation, reflecting autonomic nervous system arousal and peak mental effort.",
    "ASM": "Tracks Frontal Alpha Asymmetry, indexing approach-oriented reading motivation versus cognitive withdrawal.",
    "ThetaPower": "Monitors slow theta brain rhythms linked to working memory maintenance and mental fatigue.",
    "AlphaPower": "Captures posterior alpha oscillations, reflecting sensory gating and wakeful cognitive idling.",
    "BetaPower": "Tracks fast cortical beta activity, indicating active analytical thinking and problem-solving.",
    "Theta_Alpha_Ratio": "Measures the balance between working memory load and cortical relaxation.",
    "Theta_Beta_Ratio": "Calculates the ratio of slow theta to fast beta waves, a canonical marker of attentional control.",
    "REI": "Reading Engagement Index: synthesizes your overall mental alertness against reading cadence.",
    "CWI": "Cognitive Workload Index: reflects the total cognitive and ocular burden imposed by sentence complexity.",
    "SAI": "Sustained Attention Index: rolling 10-sentence consistency tracking your focus stability over time.",
    "MWLI_proxy": "Mental Workload Index proxy: indicates working memory buffer saturation and cognitive fatigue.",
    "VSI_proxy": "Vigilance Stability Index proxy: captures moment-to-moment stability in ocular dwell times.",
}

METRIC_SCIENTIFIC_DEFINITIONS = {
    "EEI": "A spectral band ratio calculated as Beta / (Alpha + Theta) power. Higher values indicate active cortical vigilance; lower values reflect drowsiness or mind-wandering (Pope et al., 1995).",
    "FMT": "Frontal Midline Theta (4–8 Hz) spectral power recorded at electrodes Fz and FCz (E11, E6). Synchronizes when the prefrontal cortex actively maintains multi-element context in working memory (Cavanagh & Frank, 2014).",
    "theta_fm": "Frontal Midline Theta (4–8 Hz) power at mid-frontal electrodes Fz (E11) and FCz (E6), isolating executive working memory control from global background theta.",
    "TRT": "Total Reading Time in milliseconds across all ocular fixations on a sentence, capturing lexical access, syntactic re-reading, and macro-structural comprehension (Rayner, 1998).",
    "FixationCount": "Total count of individual gaze pauses (typically 200–300 ms each) during sentence reading, indexing lexical processing effort.",
    "GD": "Gaze Duration in milliseconds: the sum of all fixations on a word during first-pass reading before moving to other words (Just & Carpenter, 1980).",
    "FFD": "First Fixation Duration in milliseconds: duration of the very first fixation on a word, reflecting early visual and orthographic identification.",
    "PupilSize": "Pupillary dilation in millimeters or z-units, driven by locus coeruleus noradrenergic activation under mental load and cognitive conflict (Beatty & Kahneman, 1966).",
    "ASM": "Frontal Alpha Asymmetry calculated as ln(Right Alpha) - ln(Left Alpha). Higher values indicate left-frontal activation associated with approach motivation and active problem-solving (Davidson, 2004).",
    "ThetaPower": "Total spectral power in the 4–8 Hz band across scalp channels, indexing cognitive workload and sustained effort.",
    "AlphaPower": "Total spectral power in the 8–13 Hz band, inversely related to active cortical sensory processing (Klimesch, 1999).",
    "BetaPower": "Total spectral power in the 13–30 Hz band, reflecting active mental concentration and task execution.",
    "Theta_Alpha_Ratio": "Ratio of 4–8 Hz to 8–13 Hz power, indexing the balance between mental exertion and sensory idling.",
    "Theta_Beta_Ratio": "Ratio of theta (4–8 Hz) to beta (13–30 Hz) power, a standard electrophysiological marker of executive control.",
    "REI": "A composite reading engagement metric combining standardized EEI, beta oscillations, and inverted theta/beta ratios.",
    "CWI": "A multi-modal composite index combining gaze dwell times, refixation frequency, and frontal midline theta power.",
    "SAI": "A 10-sentence rolling window variance formula tracking the inverse coefficient of variation of your engagement signal.",
    "MWLI_proxy": "Mental Workload Index proxy derived from the difference between reading dwell variability and cortical engagement.",
    "VSI_proxy": "Vigilance Stability Index proxy reflecting inverted rolling variability of ocular fixation times.",
}


class PersonalizedMetricSpecGenerator:
    """
    RAG-powered pedagogical specification generator.
    Couples literature from the vector store with student CSV recordings via an LLM.
    """

    def __init__(self, vector_store: Optional[HybridVectorStore] = None):
        self.rag_engine = LLMMetricRAGGenerator(vector_store=vector_store)

    def generate(
        self,
        metric_id: str,
        metric_summary: Any,
        sentence_observations: List[Any],
        df: Optional[pd.DataFrame] = None,
        synthesized_evidence: Optional[SynthesizedEvidence] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive 4-point personalized RAG specification for a metric."""
        return self.rag_engine.generate(
            metric_id=metric_id,
            metric_summary=metric_summary,
            sentence_observations=sentence_observations,
            df=df,
            synthesized_evidence=synthesized_evidence,
        )


def generate_personalized_metric_spec(
    metric_id: str,
    metric_summary: Any,
    sentence_observations: List[Any],
    df: Optional[pd.DataFrame] = None,
    synthesized_evidence: Optional[SynthesizedEvidence] = None,
    vector_store: Optional[HybridVectorStore] = None,
) -> Dict[str, Any]:
    """Helper entrypoint to generate personalized metric specifications using Pure RAG LLM."""
    engine = PersonalizedMetricSpecGenerator(vector_store=vector_store)
    return engine.generate(
        metric_id=metric_id,
        metric_summary=metric_summary,
        sentence_observations=sentence_observations,
        df=df,
        synthesized_evidence=synthesized_evidence,
    )
