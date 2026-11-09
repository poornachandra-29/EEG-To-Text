"""
Research-Aware Scientific Paper Chunker.
Splits academic texts by scientific sections (Abstract, Methodology, Findings, Discussion)
and tags chunks with physiological metrics, citation anchors, and evidence types.
"""

import re
from typing import List, Dict, Any
from neuroread_rag.discovery.models import PaperMetadata, ResearchChunk
from neuroread_rag.config import METRIC_DEFINITIONS


SECTION_PATTERNS = [
    (r"\b(abstract|summary)\b", "abstract"),
    (r"\b(methods|methodology|experimental design|materials and methods|paradigm)\b", "methodology"),
    (r"\b(results|findings|observations|data analysis)\b", "findings"),
    (r"\b(discussion|limitations|implications|conclusion)\b", "discussion"),
]


class ScientificChunker:
    """Chunks research papers preserving academic context and metric anchors."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def detect_metrics(self, text: str) -> List[str]:
        """Detect which project metrics are discussed in a passage."""
        text_lower = text.lower()
        matched = []
        for metric_id, meta in METRIC_DEFINITIONS.items():
            # Check ID exact or word boundary
            if re.search(r"\b" + re.escape(metric_id.lower()) + r"\b", text_lower):
                matched.append(metric_id)
                continue
            # Check metric keywords
            for kw in meta.get("search_keywords", []):
                if kw.lower() in text_lower:
                    matched.append(metric_id)
                    break
        return list(set(matched))

    def chunk_paper(self, paper: PaperMetadata) -> List[ResearchChunk]:
        """Convert a paper into structured, metric-aware scientific chunks."""
        chunks: List[ResearchChunk] = []

        authors_short = f"{paper.authors[0]} et al." if len(paper.authors) > 1 else (paper.authors[0] if paper.authors else "Unknown")

        # 1. Chunk Abstract
        if paper.abstract:
            abstract_text = paper.abstract.strip()
            detected_m = self.detect_metrics(abstract_text)
            # Combine detected with paper's predefined metrics
            combined_m = list(set(detected_m + paper.metrics_covered))

            chunks.append(ResearchChunk(
                chunk_id=f"{paper.paper_id}_abs_0",
                paper_id=paper.paper_id,
                paper_title=paper.title,
                doi=paper.doi,
                year=paper.year,
                authors_short=authors_short,
                section_name="abstract",
                content=abstract_text,
                metrics_tagged=combined_m,
                evidence_type="experimental" if "experiment" in abstract_text.lower() or "study" in abstract_text.lower() else "review"
            ))

        # 2. Chunk Full Text if available
        if paper.full_text:
            text = paper.full_text
            paragraphs = re.split(r"\n\s*\n", text)
            current_section = "findings"

            for p_idx, para in enumerate(paragraphs):
                para_clean = para.strip()
                if len(para_clean) < 40:
                    continue

                # Check if paragraph is a section header
                for pat, sec_name in SECTION_PATTERNS:
                    if re.match(pat, para_clean.lower()):
                        current_section = sec_name
                        break

                detected_m = self.detect_metrics(para_clean)
                chunks.append(ResearchChunk(
                    chunk_id=f"{paper.paper_id}_sec_{p_idx}",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    doi=paper.doi,
                    year=paper.year,
                    authors_short=authors_short,
                    section_name=current_section,
                    content=para_clean,
                    metrics_tagged=detected_m,
                    evidence_type="methodology" if current_section == "methodology" else "findings"
                ))

        return chunks
