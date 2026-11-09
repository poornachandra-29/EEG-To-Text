"""
Data models for the deterministic participant analysis module.
Serializes participant statistics with zero LLM hallucinations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class MetricSummary:
    """Detailed deterministic statistics for an individual physiological metric."""
    metric_id: str
    display_name: str
    category: str
    unit: str
    mean: float
    median: float
    std: float
    mad: float  # Median Absolute Deviation
    robust_sd: float  # 1.4826 * MAD
    iqr: float
    min_val: float
    max_val: float
    session_baseline: float
    personal_ref_lower: float
    personal_ref_upper: float
    relative_deviation_pct: float  # ((mean - baseline) / baseline) * 100
    deviation_category: str  # "Within Typical Range", "Elevated Above Baseline", "Below Baseline"
    variability_index: float  # std / (mean + 1e-9) or robust equivalent
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceObservation:
    """Sentence-level physiological metrics and detected patterns."""
    sentence_index: int
    sentence_text: str
    cluster_id: int
    cluster_name: str
    cluster_probability: float
    prediction_status: str
    cognitive_pattern: str
    pattern_evidence_strength: str
    top_pattern_score: float
    second_pattern_score: Optional[float]
    metrics: Dict[str, float]
    robust_z_scores: Dict[str, float]
    is_outlier: bool
    outlier_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterSummary:
    """Characteristics of each unsupervised GMM cluster for this participant."""
    cluster_id: int
    cluster_name: str
    sentence_count: int
    percentage: float
    mean_membership_probability: float
    dominant_pattern: str
    descriptive_label: str
    top_differentiating_features: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternDistribution:
    """Distribution of heuristic cognitive patterns observed across the session."""
    pattern_name: str
    sentence_count: int
    percentage: float
    evidence_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParticipantAnalysisResult:
    """Complete container for the deterministic participant analysis."""
    participant_id: str
    source_file: str
    total_sentences: int
    metrics: Dict[str, MetricSummary]
    clusters: List[ClusterSummary]
    patterns: List[PatternDistribution]
    sentence_observations: List[SentenceObservation]
    session_level_observations: List[str]
    disclaimers: List[str]
    breakdown_report: Optional[Dict[str, Any]] = None
    learning_efficiency: Optional[Dict[str, Any]] = None
    session_fatigue_curve: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
