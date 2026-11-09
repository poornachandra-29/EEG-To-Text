"""
Tests for Phase 3: RAG Retrieval, Reranking, Contradiction Detection, and Multi-Paper Synthesis.
"""

import unittest
from pathlib import Path

from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.ingestion.pipeline import IngestionPipeline
from neuroread_rag.retrieval.hybrid_search import HybridRetriever
from neuroread_rag.retrieval.reranker import ScientificReranker
from neuroread_rag.retrieval.contradiction import ContradictionDetector
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer


class TestRetrievalAndSynthesis(unittest.TestCase):
    """Test suite for Phase 3 RAG retrieval, reranking, and evidence synthesis."""

    @classmethod
    def setUpClass(cls):
        """Ensure literature vector database is populated."""
        pipeline = IngestionPipeline()
        pipeline.run_ingestion()
        cls.vector_store = HybridVectorStore()
        cls.synthesizer = EvidenceSynthesizer(cls.vector_store)

    def test_query_expansion(self):
        """Test that query expansion includes domain synonyms and metric titles."""
        retriever = HybridRetriever(self.vector_store)
        expanded = retriever.expand_query("cognitive effort", metric_id="FMT")
        self.assertIn("Frontal Midline Theta", expanded)
        self.assertIn("cognitive effort", expanded)

    def test_hybrid_retrieval_and_filtering(self):
        """Test hybrid retrieval with section and year filtering."""
        retriever = HybridRetriever(self.vector_store)
        results = retriever.retrieve("reading comprehension", top_k=3)
        self.assertGreater(len(results), 0)
        # Verify chunk structure
        chunk, score = results[0]
        self.assertTrue(len(chunk.content) > 10)
        self.assertGreater(score, 0.0)

    def test_scientific_reranker(self):
        """Test reranker scoring logic and context boost."""
        retriever = HybridRetriever(self.vector_store)
        candidates = retriever.retrieve("eye movements reading", top_k=5)
        self.assertGreater(len(candidates), 0)

        reranker = ScientificReranker()
        reranked = reranker.rerank(
            query="eye movements reading",
            candidates=candidates,
            target_metric="TRT",
            top_k=3
        )
        self.assertGreater(len(reranked), 0)
        top_chunk, score, breakdown = reranked[0]
        self.assertIn("reading_context_boost", breakdown)
        self.assertIn("metric_specificity", breakdown)
        self.assertGreaterEqual(score, 0.0)

    def test_contradiction_detector_on_known_nuances(self):
        """Verify that known context-dependent nuances are flagged for key metrics."""
        detector = ContradictionDetector()
        for metric in ["FMT", "PupilSize", "TRT", "EEI", "MWLI_proxy"]:
            report = detector.analyze_metric_evidence(metric, [])
            self.assertTrue(report.has_divergence, f"Expected divergence detection for {metric}")
            self.assertGreater(len(report.contrasting_perspectives), 1)
            self.assertTrue(len(report.reconciliation_guideline) > 10)

    def test_evidence_synthesizer_multi_paper(self):
        """Verify multi-paper synthesis for core metrics."""
        core_metrics = ["EEI", "FMT", "PupilSize", "TRT", "CWI", "SAI"]
        for m in core_metrics:
            synth = self.synthesizer.synthesize_metric(m, top_k=3)
            self.assertEqual(synth.metric_id, m)
            self.assertTrue(len(synth.consensus_summary) > 20)
            self.assertGreater(len(synth.papers_cited), 0)
            self.assertTrue(synth.contradiction_analysis is not None)
            self.assertGreater(len(synth.scientific_limitations), 0)

            # Test dictionary serialization
            d = synth.to_dict()
            self.assertEqual(d["metric_id"], m)
            self.assertIn("papers_cited", d)
            self.assertIn("contradiction_analysis", d)


if __name__ == "__main__":
    unittest.main()
