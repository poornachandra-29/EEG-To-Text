"""
Advanced Hybrid Retrieval Engine.
Integrates BM25 keyword matching, dense semantic cosine similarity,
metadata filtering, and domain-aware query expansion.
"""

from typing import List, Dict, Optional, Tuple, Any
from neuroread_rag.discovery.models import ResearchChunk
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.config import METRIC_DEFINITIONS


class HybridRetriever:
    """Retrieves relevant scientific literature using hybrid fusion and metadata filters."""

    def __init__(self, vector_store: HybridVectorStore):
        self.vector_store = vector_store

    def expand_query(self, query: str, metric_id: Optional[str] = None) -> str:
        """Expand retrieval query with domain-specific psychophysiological keywords."""
        expanded_parts = [query]
        if metric_id and metric_id in METRIC_DEFINITIONS:
            meta = METRIC_DEFINITIONS[metric_id]
            expanded_parts.append(meta.get("name", ""))
            expanded_parts.extend(meta.get("search_keywords", [])[:2])

        return " ".join(expanded_parts)

    def retrieve(
        self,
        query: str,
        metric_id: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        min_year: Optional[int] = None,
        section_filter: Optional[str] = None
    ) -> List[Tuple[ResearchChunk, float]]:
        """
        Execute hybrid retrieval with metadata filtering.
        alpha: 0.0 = pure BM25, 1.0 = pure dense, 0.5 = balanced reciprocal rank fusion.
        """
        expanded_query = self.expand_query(query, metric_id=metric_id)

        # Retrieve candidates
        candidates = self.vector_store.hybrid_search(
            query=expanded_query,
            top_k=top_k * 3,
            metric_filter=metric_id,
            alpha=alpha
        )

        filtered: List[Tuple[ResearchChunk, float]] = []
        for chunk, score in candidates:
            # Metadata filter: min_year
            if min_year and chunk.year < min_year:
                continue
            # Metadata filter: section
            if section_filter and chunk.section_name.lower() != section_filter.lower():
                continue

            filtered.append((chunk, score))
            if len(filtered) >= top_k:
                break

        return filtered
