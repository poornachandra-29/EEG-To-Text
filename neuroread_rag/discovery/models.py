"""
Data models for scientific research discovery, metadata, and chunking.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class PaperMetadata:
    """Standardized metadata for a scientific paper retrieved from academic APIs."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str  # Journal, conference, or repository
    doi: str
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    citation_count: int = 0
    sources: List[str] = field(default_factory=list)  # e.g. ["openalex", "pubmed"]
    metrics_covered: List[str] = field(default_factory=list)
    is_open_access: bool = True
    full_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchChunk:
    """Section-aware chunk of scientific text with metric and citation anchors."""
    chunk_id: str
    paper_id: str
    paper_title: str
    doi: str
    year: int
    authors_short: str
    section_name: str  # "abstract", "methodology", "findings", "discussion", "general"
    content: str
    metrics_tagged: List[str] = field(default_factory=list)
    evidence_type: str = "observational"  # "experimental", "review", "benchmark", "theoretical"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
