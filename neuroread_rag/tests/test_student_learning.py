"""
Unit tests for NeuroRead AI: Student Cognitive Learning Optimization.
Validates:
1. Learning Efficiency Score (LES, 0-100) computation and tiers.
2. Chronological Session Fatigue Curve and Pomodoro pacing interval.
3. Deterministic English Syllable & Readability (Flesch Reading Ease, Grade Level).
4. Sentence-level linguistic root cause attribution (Lexical vs Syntactic vs Conceptual Overload).
5. Student-centric reporting formats (Markdown, HTML, JSON).
"""

import unittest
import pandas as pd

from neuroread_rag.config import DEFAULT_PROFILE_CSV
from neuroread_rag.participant_analysis.sentence_comparator import (
    SentenceComparator,
    count_syllables_word,
    analyze_sentence_linguistics,
)
from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.reporting.report_generator import FinalReportGenerator
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine


class TestStudentCognitiveLearning(unittest.TestCase):
    """Test suite for Student Cognitive Learning Optimization features."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DEFAULT_PROFILE_CSV)
        cls.comparator = SentenceComparator(cls.df)
        cls.res = cls.comparator.run()
        cls.report = cls.res["report"]

        # Run full pipeline for report testing
        analyzer = ParticipantAnalyzer(DEFAULT_PROFILE_CSV)
        cls.analysis_result = analyzer.analyze()
        vector_store = HybridVectorStore()
        synthesizer = EvidenceSynthesizer(vector_store)
        engine = ResearchGroundingEngine(synthesizer)
        cls.grounded_report = engine.ground_participant_profile(cls.analysis_result)
        cls.report_generator = FinalReportGenerator()
        cls.full_report = cls.report_generator.generate_full_report(cls.grounded_report, cls.analysis_result)

    def test_syllable_and_readability_metrics(self):
        """Verify deterministic English syllable counting and readability formulas."""
        self.assertEqual(count_syllables_word("the"), 1)
        self.assertEqual(count_syllables_word("reading"), 2)
        self.assertEqual(count_syllables_word("cognitive"), 3)
        self.assertEqual(count_syllables_word("neurophysiological"), 7)

        simple_text = "The cat sat on the mat."
        simple_res = analyze_sentence_linguistics(simple_text)
        self.assertGreater(simple_res["flesch_reading_ease"], 80.0)
        self.assertLess(simple_res["grade_level"], 5.0)
        self.assertEqual(simple_res["complex_words"], 0)

        complex_text = (
            "Neurophysiological investigation demonstrates that simultaneous electroencephalography "
            "and pupillometry quantify instantaneous working memory cognitive workload saturation."
        )
        complex_res = analyze_sentence_linguistics(complex_text)
        self.assertGreater(complex_res["complex_words_pct"], 30.0)
        self.assertGreater(complex_res["grade_level"], 12.0)

    def test_learning_efficiency_score(self):
        """Verify Learning Efficiency Score (LES) bounds, factors, and tier categorization."""
        self.assertIn("learning_efficiency", self.report)
        les = self.report["learning_efficiency"]

        score = les["score"]
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

        self.assertIn("tier", les)
        self.assertIn("flow_factor", les)
        self.assertIn("cadence_stability_factor", les)
        self.assertIn("workload_ease_factor", les)

        # Check factor validity
        self.assertGreaterEqual(les["flow_factor"], 0.0)
        self.assertLessEqual(les["flow_factor"], 100.0)
        self.assertGreaterEqual(les["cadence_stability_factor"], 0.0)
        self.assertLessEqual(les["cadence_stability_factor"], 100.0)

    def test_session_fatigue_curve(self):
        """Verify session fatigue curve chronological checkpoints and optimal study block."""
        self.assertIn("session_fatigue_curve", self.report)
        fatigue = self.report["session_fatigue_curve"]

        self.assertIn("checkpoints", fatigue)
        self.assertGreaterEqual(len(fatigue["checkpoints"]), 3)

        for cp in fatigue["checkpoints"]:
            self.assertIn("sentence_range", cp)
            self.assertIn("mean_eei", cp)
            self.assertIn("status", cp)

        self.assertIn("fatigue_onset_sentence", fatigue)
        self.assertGreater(fatigue["fatigue_onset_sentence"], 0)

        pomo = fatigue["recommended_study_block_minutes"]
        self.assertGreaterEqual(pomo, 15)
        self.assertLessEqual(pomo, 40)

    def test_sentence_breakdown_linguistics(self):
        """Verify top flow and breakdown sentences include readability & root cause metadata."""
        top_flow = self.report["top_flow_sentences"]
        self.assertGreater(len(top_flow), 0)
        for s in top_flow:
            self.assertIn("flesch_reading_ease", s)
            self.assertIn("grade_level", s)
            self.assertIn("linguistic_root_cause", s)

        top_breakdown = self.report["top_breakdown_sentences"]
        self.assertGreater(len(top_breakdown), 0)
        for s in top_breakdown:
            self.assertIn("flesch_reading_ease", s)
            self.assertIn("grade_level", s)
            self.assertIn("linguistic_root_cause", s)
            # Ensure root cause is meaningful (e.g. Lexical Overload or Syntactic Overload)
            self.assertTrue(len(s["linguistic_root_cause"]) > 5)

    def test_student_report_formatting(self):
        """Verify report generator includes LES, fatigue curve, and student terminology."""
        self.assertIn("learning_efficiency", self.full_report)
        self.assertIn("session_fatigue_curve", self.full_report)

        # Verify Markdown export contains student headers & LES
        md = self.report_generator.generate_markdown_report(self.full_report)
        self.assertIn("Student Cognitive Learning Assessment", md)
        self.assertIn("Learning Efficiency Score", md)
        self.assertIn("Flow State", md)

        # Verify HTML export contains LES hero card & student badge
        html = self.report_generator.generate_html_report(self.full_report)
        self.assertIn("Student Cognitive Learning Assessment", html)
        self.assertIn("Learning Efficiency Score", html)
        self.assertIn("Flow Factor", html)


if __name__ == "__main__":
    unittest.main()
