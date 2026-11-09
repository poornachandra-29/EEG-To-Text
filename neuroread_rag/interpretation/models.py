"""
Data models for research-grounded interpretation of participant physiological data.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ResearchFindingItem:
    """Scientific finding from a specific paper mapped to participant observation."""
    paper_title: str
    authors: str
    year: int
    doi: str
    venue: str
    claim: str
    experimental_context: str
    evidence_strength: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricInterpretation:
    """Rigorous research-grounded interpretation of a single physiological metric."""
    metric_id: str
    metric_name: str
    category: str
    unit: str
    scientific_definition: str
    participant_measurement: Dict[str, Any]
    personal_baseline_comparison: Dict[str, Any]
    sentence_level_dynamics: Dict[str, Any]
    retrieved_scientific_findings: List[ResearchFindingItem]
    research_compatibility: str
    divergence_or_nuance: Dict[str, Any]
    evidence_strength: str
    interpretation_and_limitations: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["retrieved_scientific_findings"] = [f.to_dict() for f in self.retrieved_scientific_findings]
        return res


@dataclass
class ParticipantGroundedReport:
    """Comprehensive research-grounded report covering all physiological domains."""
    participant_id: str
    total_sentences: int
    metric_interpretations: Dict[str, MetricInterpretation]
    eeg_spectral_summary: str
    eye_tracking_summary: str
    cognitive_workload_summary: str
    focus_attention_summary: str
    gmm_cluster_research_grounding: List[Dict[str, Any]]
    mandatory_disclaimers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["metric_interpretations"] = {k: v.to_dict() for k, v in self.metric_interpretations.items()}
        return res
