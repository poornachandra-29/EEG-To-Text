"""
Metric-Specific Scientific Evidence Extractor.
Extracts empirical findings, experimental contexts, and limitations from ingested literature.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from neuroread_rag.discovery.models import ResearchChunk
from neuroread_rag.config import METRIC_DEFINITIONS


@dataclass
class MetricEvidence:
    """Structured empirical finding extracted from scientific literature."""
    metric_id: str
    paper_title: str
    authors_short: str
    year: int
    doi: str
    venue: str
    claim: str
    directionality: str  # "Increases with mental workload", "Suppressed during engagement", etc.
    experimental_context: str  # "Natural reading (ZuCo)", "Working memory task", "Continuous performance"
    evidence_strength: str  # "High (Seminal Peer-Reviewed)", "Moderate", "Preliminary"
    limitations: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceExtractor:
    """Extracts metric-specific scientific claims and contexts from research chunks."""

    DIRECTION_PATTERNS = [
        (r"\b(increase[s|d]?|elevat[e|ed|ion]|greater|higher|augment[ed]?)\b.*\b(workload|effort|difficulty|demand)\b", "Increases with cognitive workload / mental effort"),
        (r"\b(decrease[s|d]?|depress[ed]?|lower|diminish[ed]?|reduc[e|ed|tion])\b.*\b(fatigue|drift|lapse|disengagement)\b", "Depressed during fatigue or attentional disengagement"),
        (r"\b(desynchroniz|suppress[ed|ion]|decrease)\b.*\b(alpha)\b", "Alpha desynchronization / suppression during active attention"),
        (r"\b(theta)\b.*\b(increase[s|d]?|elevat[e|ed|ion]|synchroniz)\b", "Theta power increases with memory load and conflict monitoring"),
        (r"\b(pupil|dilation)\b.*\b(increase|larger|dilate)\b", "Pupil dilation increases with autonomic arousal and processing load"),
        (r"\b(longer|prolonged|elevated)\b.*\b(gaze|reading time|fixation)\b", "Prolonged fixation/reading time indicates lexical or syntactic processing difficulty"),
        (r"\b(variabilit|erratic|fluctuat)\b", "Increased variability reflects attentional instability or cadence drift"),
    ]

    def extract_evidence(self, chunk: ResearchChunk) -> List[MetricEvidence]:
        """Parse chunk sentences to identify metric claims, directionality, and experimental contexts."""
        evidence_list: List[MetricEvidence] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk.content)

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 30:
                continue

            for m in chunk.metrics_tagged:
                meta = METRIC_DEFINITIONS.get(m, {})
                m_name = meta.get("name", m)
                m_keywords = meta.get("search_keywords", [])

                # Check if sentence mentions metric
                s_lower = s_clean.lower()
                is_mentioned = (
                    re.search(r"\b" + re.escape(m.lower()) + r"\b", s_lower) is not None or
                    any(kw.lower() in s_lower for kw in m_keywords)
                )

                if is_mentioned:
                    # Determine directionality
                    direction = "Contextual physiological response"
                    for pat, dir_label in self.DIRECTION_PATTERNS:
                        if re.search(pat, s_lower):
                            direction = dir_label
                            break

                    # Determine context
                    context = "Reading and cognitive task paradigms"
                    if "zuco" in s_lower or "reading" in s_lower:
                        context = "Natural reading task (sentences with concurrent EEG & eye tracking)"
                    elif "flight" in s_lower or "simulation" in s_lower:
                        context = "Continuous high-workload task simulation"
                    elif "memory" in s_lower:
                        context = "Working memory and executive control task"

                    # Determine evidence strength
                    strength = "High (Peer-Reviewed Canonical Study)" if chunk.year < 2022 else "Moderate"

                    # Determine scientific limitations
                    limitations = "Correlational physiological observation; must be evaluated relative to personal baseline."
                    if m in ["MWLI_proxy", "SAI", "VSI_proxy"]:
                        limitations = "Heuristic composite proxy; does not establish verified psychological state or clinical disorder."
                    elif m in ["EEI", "CWI"]:
                        limitations = "Exploratory composite index; task and lighting conditions influence baseline."

                    evidence_list.append(MetricEvidence(
                        metric_id=m,
                        paper_title=chunk.paper_title,
                        authors_short=chunk.authors_short,
                        year=chunk.year,
                        doi=chunk.doi,
                        venue=chunk.section_name.title(),
                        claim=s_clean,
                        directionality=direction,
                        experimental_context=context,
                        evidence_strength=strength,
                        limitations=limitations
                    ))
                    break  # One evidence item per sentence per chunk

        return evidence_list
