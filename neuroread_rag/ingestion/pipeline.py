"""
Ingestion Pipeline Coordinator.
Orchestrates discovery, section-aware chunking, evidence extraction, and vector indexing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from neuroread_rag.discovery.aggregator import ResearchDiscoveryAggregator
from neuroread_rag.ingestion.chunker import ScientificChunker
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.ingestion.evidence_extractor import EvidenceExtractor, MetricEvidence
from neuroread_rag.config import OUTPUT_DIR, METRIC_DEFINITIONS


class IngestionPipeline:
    """End-to-end research ingestion and index construction."""

    def __init__(self, db_path: Optional[Path] = None):
        self.aggregator = ResearchDiscoveryAggregator()
        self.chunker = ScientificChunker()
        self.vector_store = HybridVectorStore(db_path=db_path)
        self.evidence_extractor = EvidenceExtractor()

    def run_ingestion(self) -> Dict[str, Any]:
        """Execute full discovery and ingestion process across all metrics."""
        print("[IngestionPipeline] Starting multi-source paper discovery...")
        all_papers_dict = self.aggregator.discover_all_metrics()

        # Deduplicate all papers across metrics
        unique_papers = {}
        for metric, papers in all_papers_dict.items():
            for p in papers:
                key = p.doi.lower() if p.doi else p.title.lower()
                if key not in unique_papers:
                    unique_papers[key] = p
                else:
                    for m in p.metrics_covered:
                        if m not in unique_papers[key].metrics_covered:
                            unique_papers[key].metrics_covered.append(m)

        papers_list = list(unique_papers.values())
        print(f"[IngestionPipeline] Ingesting {len(papers_list)} unique papers...")

        # Chunk papers
        all_chunks = []
        for paper in papers_list:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)

        print(f"[IngestionPipeline] Created {len(all_chunks)} section-aware scientific chunks.")

        # Extract metric evidence
        all_evidence: List[MetricEvidence] = []
        for chunk in all_chunks:
            evs = self.evidence_extractor.extract_evidence(chunk)
            all_evidence.extend(evs)

        print(f"[IngestionPipeline] Extracted {len(all_evidence)} metric-specific evidence claims.")

        # Index papers and chunks into SQLite and BM25/Vector store
        self.vector_store.index_papers(papers_list, all_chunks)

        # Save evidence catalog to JSON
        evidence_file = OUTPUT_DIR / "extracted_evidence.json"
        with open(evidence_file, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in all_evidence], f, indent=2)

        return {
            "papers_count": len(papers_list),
            "chunks_count": len(all_chunks),
            "evidence_count": len(all_evidence),
            "db_path": str(self.vector_store.db_path),
            "evidence_path": str(evidence_file)
        }
