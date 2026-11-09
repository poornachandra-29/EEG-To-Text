"""
Scientific Evidence Reranker.
Reranks retrieved research chunks by combining relevance, citation authority,
experimental paradigm alignment, and exact metric specificity.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from neuroread_rag.discovery.models import ResearchChunk


class ScientificReranker:
    """Reranks retrieved research candidates using multi-factor academic scoring."""

    def __init__(
        self,
        relevance_weight: float = 0.50,
        reading_context_boost: float = 0.20,
        metric_specificity_boost: float = 0.20,
        authority_weight: float = 0.10
    ):
        self.relevance_weight = relevance_weight
        self.reading_context_boost = reading_context_boost
        self.metric_specificity_boost = metric_specificity_boost
        self.authority_weight = authority_weight

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[ResearchChunk, float]],
        target_metric: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[ResearchChunk, float, Dict[str, float]]]:
        """
        Rerank candidate chunks and return (chunk, final_score, score_breakdown).
        """
        if not candidates:
            return []

        # Find max base score for normalization
        max_base = max(score for _, score in candidates) if candidates else 1.0
        if max_base < 1e-9:
            max_base = 1.0

        reranked = []

        for chunk, base_score in candidates:
            norm_rel = base_score / max_base

            # 1. Reading Context Boost (ZuCo / natural reading / sentence comprehension)
            content_lower = chunk.content.lower()
            reading_score = 0.0
            if "natural reading" in content_lower or "zuco" in content_lower:
                reading_score = 1.0
            elif "reading" in content_lower or "sentence" in content_lower:
                reading_score = 0.7
            elif "eye-tracking" in content_lower or "eeg" in content_lower:
                reading_score = 0.4

            # 2. Metric Specificity Boost
            metric_score = 0.0
            if target_metric:
                if target_metric in chunk.metrics_tagged:
                    metric_score = 1.0
                elif target_metric.lower() in content_lower:
                    metric_score = 0.8

            # 3. Citation Authority / Canonical Weight
            # Normalize year: seminal older papers (>1000 citations) and recent high quality
            authority_score = 0.8 if chunk.year <= 2018 else 0.6

            # Combined weighted score
            final_score = (
                (self.relevance_weight * norm_rel) +
                (self.reading_context_boost * reading_score) +
                (self.metric_specificity_boost * metric_score) +
                (self.authority_weight * authority_score)
            )

            breakdown = {
                "base_relevance": round(norm_rel, 3),
                "reading_context_boost": round(reading_score, 3),
                "metric_specificity": round(metric_score, 3),
                "authority_score": round(authority_score, 3),
                "final_score": round(final_score, 4)
            }

            reranked.append((chunk, final_score, breakdown))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
