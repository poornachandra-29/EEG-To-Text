"""
Tests for Phase 5: Recommendation Engine and 10-Section Final Report Generator.
"""

import unittest
from pathlib import Path

from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine
from neuroread_rag.recommendations.recommender import RecommendationEngine
from neuroread_rag.reporting.report_generator import FinalReportGenerator
from neuroread_rag.config import DEFAULT_PROFILE_CSV


class TestRecommendationsAndReport(unittest.TestCase):
    """Test suite for Phase 5 recommendations and report generation."""

    @classmethod
    def setUpClass(cls):
        analyzer = ParticipantAnalyzer(DEFAULT_PROFILE_CSV)
        cls.analysis_result = analyzer.analyze()
        vector_store = HybridVectorStore()
        synthesizer = EvidenceSynthesizer(vector_store)
        engine = ResearchGroundingEngine(synthesizer)
        cls.grounded_report = engine.ground_participant_profile(cls.analysis_result)
        cls.recommender = RecommendationEngine()
        cls.report_generator = FinalReportGenerator()

    def test_recommendation_generation(self):
        """Verify that recommendations are personalized and literature-backed."""
        recs = self.recommender.generate_recommendations(self.grounded_report, self.analysis_result)
        self.assertGreater(len(recs), 0)

        for r in recs:
            self.assertTrue(len(r.title) > 5)
            self.assertTrue(len(r.observed_basis) > 10)
            self.assertTrue(len(r.actionable_guidance) > 15)
            self.assertTrue(len(r.scientific_rationale) > 20)
            self.assertGreater(len(r.supporting_citations), 0)
            for c in r.supporting_citations:
                self.assertTrue(len(c.doi) > 3)
                self.assertTrue(len(c.citation_text) > 3)

    def test_full_10_section_report_structure(self):
        """Verify that all 10 required sections exist in the compiled report data."""
        data = self.report_generator.generate_full_report(self.grounded_report, self.analysis_result)

        required_sections = [
            "section_1_executive_summary",
            "section_2_focus_attention",
            "section_3_eeg_analysis",
            "section_4_eye_tracking_analysis",
            "section_5_cognitive_workload",
            "section_6_sentence_observations",
            "section_7_research_comparison",
            "section_8_recommendations",
            "section_9_scientific_limitations",
            "section_10_references"
        ]

        for sec in required_sections:
            self.assertIn(sec, data, f"Missing required report section: {sec}")

        # Check references count
        self.assertGreater(len(data["section_10_references"]), 0)
        # Check comparison count
        self.assertGreater(len(data["section_7_research_comparison"]), 0)

    def test_report_export_formats(self):
        """Verify report generation in JSON, Markdown, and HTML formats."""
        out_paths = self.report_generator.save_reports(self.grounded_report, self.analysis_result)

        self.assertTrue(out_paths["json"].exists())
        self.assertTrue(out_paths["markdown"].exists())
        self.assertTrue(out_paths["html"].exists())

        # Verify Markdown content
        md_text = out_paths["markdown"].read_text(encoding="utf-8")
        self.assertIn("# NeuroRead AI", md_text)
        self.assertIn("## 1. Executive Summary", md_text)
        self.assertIn("## 10. Complete Scientific References", md_text)

        # Verify HTML content
        html_text = out_paths["html"].read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_text)
        self.assertIn("Executive Summary", html_text)
        self.assertIn("Print / Save as PDF", html_text)


if __name__ == "__main__":
    unittest.main()
