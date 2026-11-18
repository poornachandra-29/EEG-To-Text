"""
Typed Schemas for NeuroRead RAG Pipeline Stages.
Defines explicit data contracts passed between pipeline stages.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class SentenceFeatures(BaseModel):
    """Extracted sentence-level features and metadata."""
    sentence_index: int
    sentence_text: str
    patient_id: str
    task_name: str
    eeg_duration_s: float = 0.0
    eeg_reliability: str = "full"  # "full", "low", "insufficient"
    artifact_fraction: float = 0.0
    
    # EEG features
    EEI: Optional[float] = None
    FMT: Optional[float] = None  # Legacy or primary frontal theta
    theta_global: Optional[float] = None
    theta_fm: Optional[float] = None
    ASM: Optional[float] = None
    ThetaPower: Optional[float] = None
    AlphaPower: Optional[float] = None
    BetaPower: Optional[float] = None
    Theta_Alpha_Ratio: Optional[float] = None
    Theta_Beta_Ratio: Optional[float] = None
    
    # Eye-tracking features
    GD: Optional[float] = None
    TRT: Optional[float] = None
    FFD: Optional[float] = None
    PupilSize: Optional[float] = None
    FixationCount: Optional[float] = None
    RegressionCount: Optional[float] = 0.0
    
    # Text covariates
    n_words: Optional[int] = None
    mean_word_len: Optional[float] = None
    max_word_len: Optional[int] = None
    mean_zipf: Optional[float] = None
    min_zipf: Optional[float] = None
    total_surprisal: Optional[float] = None
    mean_surprisal: Optional[float] = None
    max_surprisal: Optional[float] = None
    max_dep_depth: Optional[int] = None
    n_clauses: Optional[int] = None
    n_entities_numerals: Optional[int] = None
    flesch_ease: Optional[float] = None
    fk_grade: Optional[float] = None


class SentenceLabels(BaseModel):
    """Assigned labels and diagnostic categories for a sentence."""
    sentence_index: int
    breakdown_behav: Optional[int] = 0  # 1 if behavioral breakdown, 0 otherwise
    excess_processing: Optional[float] = 0.0
    comprehension_error: Optional[float] = None  # 1 if error, 0 if correct, None if unqueried
    legacy_state: Optional[str] = None  # "🟢 Flow State", "🔴 Comprehension Breakdown", etc.
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    cluster_confidence: Optional[float] = None


class Finding(BaseModel):
    """Diagnostic finding derived from reader analytics."""
    id: str
    type: str  # "lexical_overload", "syntactic_overload", "fatigue_onset", etc.
    severity: str  # "high", "moderate", "low"
    sentence_ids: List[int] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    reliability: str = "high"


class Recommendation(BaseModel):
    """Personalized learning intervention."""
    id: str
    title: str
    category: str
    observed_basis: str
    actionable_guidance: str
    scientific_rationale: str
    supporting_citations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_strength: str = "direct"


class Report(BaseModel):
    """End-to-end participant diagnostic report."""
    metadata: Dict[str, Any]
    learning_efficiency: Dict[str, Any]
    session_fatigue_curve: Dict[str, Any]
    findings: List[Finding] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    sections: Dict[str, Any] = Field(default_factory=dict)
