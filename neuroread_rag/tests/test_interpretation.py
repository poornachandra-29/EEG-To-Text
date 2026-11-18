"""
Tests for Phase 4: Numerical Analysis and Research Grounding Integration.
"""

import unittest
from pathlib import Path

from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.guardrails import ScientificGuardrailValidator
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine
from neuroread_rag.config import DEFAULT_PROFILE_CSV


class TestInterpretationAndGrounding(unittest.TestCase):
    """Test suite for research grounding and non-diagnostic guardrails."""

    @classmethod
    def setUpClass(cls):
        analyzer = ParticipantAnalyzer(DEFAULT_PROFILE_CSV)
        cls.participant_result = analyzer.analyze()
        vector_store = HybridVectorStore()
        cls.synthesizer = EvidenceSynthesizer(vector_store)
        cls.engine = ResearchGroundingEngine(cls.synthesizer)

    def test_guardrails_catches_diagnostic_overreach(self):
        """Verify that guardrails catch forbidden diagnostic language."""
        validator = ScientificGuardrailValidator()

        clean_text = "The participant showed elevated Frontal Midline Theta consistent with working memory effort."
        is_valid, violations = validator.validate_text(clean_text)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

        diagnostic_text = "The participant is diagnosed with ADHD and clinical impairment based on EEG."
        is_valid, violations = validator.validate_text(diagnostic_text)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)

    def test_grounding_preserves_deterministic_numbers(self):
        """Verify that participant measurements in the grounded report match deterministic stats exactly."""
        report = self.engine.ground_participant_profile(self.participant_result)
        self.assertEqual(report.participant_id, "resultsZPH_NR")
        self.assertEqual(report.total_sentences, 250)

        # Check EEI measurements
        self.assertIn("EEI", report.metric_interpretations)
        eei_interp = report.metric_interpretations["EEI"]
        self.assertEqual(eei_interp.participant_measurement["mean"], self.participant_result.metrics["EEI"].mean)
        self.assertEqual(eei_interp.participant_measurement["median"], self.participant_result.metrics["EEI"].median)
        self.assertEqual(eei_interp.personal_baseline_comparison["session_baseline"], self.participant_result.metrics["EEI"].session_baseline)
        self.assertEqual(eei_interp.personal_baseline_comparison["deviation_category"], self.participant_result.metrics["EEI"].deviation_category)

    def test_metric_interpretation_structure(self):
        """Verify all required sections exist in metric interpretations."""
        report = self.engine.ground_participant_profile(self.participant_result)

        for m_id, interp in report.metric_interpretations.items():
            self.assertTrue(len(interp.scientific_definition) > 10)
            self.assertIn("mean", interp.participant_measurement)
            self.assertIn("session_baseline", interp.personal_baseline_comparison)
            self.assertIn("variability_index", interp.sentence_level_dynamics)
            self.assertTrue(len(interp.research_compatibility) > 20)
            self.assertTrue(len(interp.interpretation_and_limitations) > 20)
            self.assertIn("has_divergence", interp.divergence_or_nuance)

    def test_domain_syntheses_and_cluster_grounding(self):
        """Verify domain level summaries and cluster grounding."""
        report = self.engine.ground_participant_profile(self.participant_result)
        self.assertTrue("EEG" in report.eeg_spectral_summary)
        self.assertTrue("Eye-Tracking" in report.eye_tracking_summary)
        self.assertTrue("Workload" in report.cognitive_workload_summary)
        self.assertTrue("Focus" in report.focus_attention_summary)
        self.assertGreater(len(report.gmm_cluster_research_grounding), 0)

    def test_save_grounded_report(self):
        """Verify saving grounded report to JSON."""
        report = self.engine.ground_participant_profile(self.participant_result)
        out_file = self.engine.save_grounded_report(report)
        self.assertTrue(out_file.exists())
        self.assertGreater(out_file.stat().st_size, 5000)

    def test_personalized_metric_specifications(self):
        """Verify that every metric includes a 6-dimension personalized pedagogical specification."""
        report = self.engine.ground_participant_profile(self.participant_result)
        
        required_keys = [
            "metric_id",
            "what_does_it_do",
            "what_is_it_actually",
            "what_it_says_about_you",
            "readability_connection",
            "where_did_you_make_a_mistake",
            "how_to_resolve",
            "what_this_metric_says",
            "how_you_are_performing",
            "how_to_improve",
            "what_improving_means_to_you",
        ]
        
        for m_id in ["EEI", "FMT", "TRT", "FixationCount", "BetaPower"]:
            self.assertIn(m_id, report.metric_interpretations)
            interp = report.metric_interpretations[m_id]
            spec = interp.personalized_specification
            self.assertIsNotNone(spec, f"Metric {m_id} is missing personalized_specification")
            
            for k in required_keys:
                self.assertIn(k, spec, f"Metric {m_id} missing {k} in personalized_specification")
                
            self.assertGreater(len(spec["what_does_it_do"]), 10)
            self.assertGreater(len(spec["what_is_it_actually"]), 10)
            self.assertGreater(len(spec["what_it_says_about_you"]), 10)
            self.assertGreater(len(spec["readability_connection"]), 10)
            self.assertGreater(len(spec["what_this_metric_says"]), 20)
            self.assertGreater(len(spec["how_you_are_performing"]), 20)
            self.assertGreater(len(spec["what_improving_means_to_you"]), 20)
            self.assertGreaterEqual(len(spec["how_to_improve"]), 1)
            
            diag = spec["where_did_you_make_a_mistake"]
            self.assertIsInstance(diag, dict)
            self.assertIn("sentence_index", diag)
            self.assertIn("sentence_text", diag)
            self.assertIn("grade_level", diag)
            self.assertIn("flesch_reading_ease", diag)
            self.assertIn("diagnosis_note", diag)
            self.assertGreater(len(diag["diagnosis_note"]), 10)
            
            resolutions = spec["how_to_resolve"]
            self.assertIsInstance(resolutions, list)
            self.assertGreaterEqual(len(resolutions), 1)
            for res in resolutions:
                self.assertGreater(len(res), 15)


if __name__ == "__main__":
    unittest.main()
