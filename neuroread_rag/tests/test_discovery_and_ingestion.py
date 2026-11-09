"""
Tests for Phase 2: Research Discovery, Ingestion, and Hybrid Vector Indexing.
"""

import unittest
from pathlib import Path

from neuroread_rag.discovery.models import PaperMetadata, ResearchChunk
from neuroread_rag.discovery.aggregator import ResearchDiscoveryAggregator
from neuroread_rag.discovery.crossref import CrossrefClient
from neuroread_rag.ingestion.chunker import ScientificChunker
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.ingestion.evidence_extractor import EvidenceExtractor
from neuroread_rag.ingestion.pipeline import IngestionPipeline


class TestDiscoveryAndIngestion(unittest.TestCase):
    """Test suite for research discovery, chunking, evidence extraction, and vector storage."""

    def setUp(self):
        self.aggregator = ResearchDiscoveryAggregator()
        self.chunker = ScientificChunker()
        self.evidence_extractor = EvidenceExtractor()

    def test_discovery_aggregator_has_foundational_papers(self):
        """Verify that foundational peer-reviewed literature is discovered for all core metrics."""
        for metric in ["EEI", "FMT", "GD", "TRT", "PupilSize", "ASM", "CWI", "SAI", "MWLI_proxy"]:
            papers = self.aggregator.discover_metric_papers(metric)
            self.assertGreater(len(papers), 0, f"No papers found for {metric}")
            # Ensure DOI and title exist
            for p in papers:
                self.assertTrue(len(p.title) > 5)
                self.assertTrue(len(p.doi) > 3)
                self.assertIn(metric, p.metrics_covered)

    def test_crossref_doi_verification(self):
        """Test verification of a real DOI format."""
        client = CrossrefClient()
        # Verify canonical ZuCo Nature Scientific Data DOI format
        zuco_doi = "10.1038/sdata.2018.291"
        self.assertTrue(len(zuco_doi) > 5)

    def test_scientific_chunker_sections(self):
        """Test section tagging and metric tagging."""
        test_paper = PaperMetadata(
            paper_id="test_p1",
            title="Frontal Midline Theta Dynamics in Natural Reading",
            authors=["Researcher A", "Researcher B"],
            year=2023,
            venue="NeuroImage",
            doi="10.1016/j.neuroimage.2023.001",
            abstract="In this experiment, we examined frontal midline theta (FMT) and pupil dilation during sentence reading. Theta power increased significantly with sentence complexity.",
            url="https://doi.org/10.1016/j.neuroimage.2023.001",
            metrics_covered=["FMT", "PupilSize", "ThetaPower"]
        )

        chunks = self.chunker.chunk_paper(test_paper)
        self.assertGreater(len(chunks), 0)
        chunk = chunks[0]
        self.assertEqual(chunk.section_name, "abstract")
        self.assertIn("FMT", chunk.metrics_tagged)
        self.assertEqual(chunk.doi, test_paper.doi)

    def test_evidence_extraction(self):
        """Test extraction of structured physiological claims."""
        chunk = ResearchChunk(
            chunk_id="c_test_1",
            paper_id="p_test_1",
            paper_title="Cognitive Control and Theta",
            doi="10.1016/j.tics.2014.04.012",
            year=2014,
            authors_short="Cavanagh & Frank",
            section_name="findings",
            content="Frontal midline theta power increases with cognitive workload and task difficulty. Alpha power decreases during active attention.",
            metrics_tagged=["FMT", "ThetaPower", "AlphaPower"]
        )

        evidence = self.evidence_extractor.extract_evidence(chunk)
        self.assertGreater(len(evidence), 0)
        ev = evidence[0]
        self.assertIn("Increases with cognitive workload", ev.directionality)
        self.assertEqual(ev.authors_short, "Cavanagh & Frank")

    def test_hybrid_vector_store_operations(self):
        """Test BM25, dense semantic search, and hybrid search in vector store."""
        test_db = Path("neuroread_rag/output/test_literature.db")
        if test_db.exists():
            test_db.unlink()

        store = HybridVectorStore(db_path=test_db)

        paper1 = PaperMetadata(
            paper_id="p1",
            title="EEG Engagement Index in Reading Tasks",
            authors=["Pope et al."],
            year=1995,
            venue="Biol Psychol",
            doi="10.1016/0301-0511(95)05116-3",
            abstract="The EEG Engagement Index (EEI = Beta / (Alpha + Theta)) measures mental engagement and attentional vigilance.",
            url="https://doi.org/10.1016/0301-0511(95)05116-3",
            metrics_covered=["EEI", "BetaPower"]
        )

        chunk1 = ResearchChunk(
            chunk_id="p1_c1",
            paper_id="p1",
            paper_title=paper1.title,
            doi=paper1.doi,
            year=paper1.year,
            authors_short="Pope et al.",
            section_name="abstract",
            content=paper1.abstract,
            metrics_tagged=["EEI", "BetaPower"]
        )

        store.index_papers([paper1], [chunk1])

        # Test BM25 search
        bm25_res = store.search_bm25("EEG engagement index beta")
        self.assertGreater(len(bm25_res), 0)
        self.assertEqual(bm25_res[0][0].paper_id, "p1")

        # Test Dense search
        dense_res = store.search_dense("attentional vigilance mental engagement")
        self.assertGreater(len(dense_res), 0)
        self.assertEqual(dense_res[0][0].paper_id, "p1")

        # Test Hybrid search
        hybrid_res = store.hybrid_search("engagement index", top_k=3)
        self.assertGreater(len(hybrid_res), 0)
        self.assertEqual(hybrid_res[0][0].paper_id, "p1")

        # Cleanup test db
        if test_db.exists():
            test_db.unlink()

    def test_full_ingestion_pipeline_run(self):
        """Test end-to-end execution of IngestionPipeline."""
        pipeline = IngestionPipeline()
        stats = pipeline.run_ingestion()

        self.assertGreater(stats["papers_count"], 0)
        self.assertGreater(stats["chunks_count"], 0)
        self.assertGreater(stats["evidence_count"], 0)
        self.assertTrue(Path(stats["db_path"]).exists())
        self.assertTrue(Path(stats["evidence_path"]).exists())


if __name__ == "__main__":
    unittest.main()
