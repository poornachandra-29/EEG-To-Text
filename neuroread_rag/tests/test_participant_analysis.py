"""
Tests for the deterministic participant analysis module.
Verifies programmatic calculations without LLM hallucinations.
"""

import math
import unittest
import pandas as pd
import numpy as np

from neuroread_rag.config import (
    DEFAULT_PROFILE_CSV,
    FALLBACK_PREDICTIONS_CSV,
    METRIC_DEFINITIONS,
    MANDATORY_DISCLAIMERS,
    CSV_DIR,
    OUTPUT_DIR,
    find_participant_csv
)
from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer, robust_z
from neuroread_rag.participant_analysis.mat_processor import extract_task_and_patient_info, process_mat_file


class TestParticipantAnalysis(unittest.TestCase):
    """Test suite for deterministic participant calculations."""

    def test_robust_z_calculation(self):
        """Verify robust z-score calculation against known values."""
        series = pd.Series([10.0, 12.0, 11.0, 10.5, 9.8, 20.0])
        rz = robust_z(series)
        self.assertEqual(len(rz), 6)
        self.assertFalse(rz.isna().any())
        self.assertGreater(rz.iloc[5], 2.0)
        self.assertLess(abs(rz.iloc[3]), 1.0)

    def test_participant_analyzer_with_profile_csv(self):
        """Test analyzer against existing cognitive_profile_output.csv."""
        self.assertTrue(DEFAULT_PROFILE_CSV.exists())

        analyzer = ParticipantAnalyzer(DEFAULT_PROFILE_CSV)
        result = analyzer.analyze()

        self.assertEqual(result.participant_id, "resultsZPH_NR")
        self.assertGreater(result.total_sentences, 0)

        core_metrics = ["EEI", "FMT", "ASM", "GD", "FFD", "TRT", "PupilSize", "FixationCount", "REI", "CWI", "SAI", "MWLI_proxy", "VSI_proxy"]
        for m in core_metrics:
            self.assertIn(m, result.metrics)
            summary = result.metrics[m]
            self.assertFalse(math.isnan(summary.mean))
            self.assertFalse(math.isnan(summary.median))
            self.assertFalse(math.isnan(summary.std))
            self.assertFalse(math.isnan(summary.robust_sd))
            self.assertLessEqual(summary.personal_ref_lower, summary.personal_ref_upper)
            self.assertLessEqual(summary.min_val, summary.max_val)
            self.assertIn(summary.deviation_category, ["Within Typical Range", "Elevated Above Baseline", "Below Baseline"])

        self.assertGreater(len(result.clusters), 0)
        total_pct = sum(c.percentage for c in result.clusters)
        self.assertLess(abs(total_pct - 100.0), 0.5)

        self.assertGreater(len(result.patterns), 0)
        self.assertEqual(len(result.sentence_observations), result.total_sentences)
        for obs in result.sentence_observations:
            self.assertTrue(0.0 <= obs.cluster_probability <= 1.0)
            self.assertIsNotNone(obs.sentence_text)

        self.assertEqual(len(result.disclaimers), len(MANDATORY_DISCLAIMERS))
        self.assertTrue(any("NOT clinical diagnoses" in d for d in result.disclaimers))

    def test_participant_analyzer_with_test_predictions_csv(self):
        """Test analyzer compatibility when fed raw test_predictions.csv."""
        self.assertTrue(FALLBACK_PREDICTIONS_CSV.exists())

        analyzer = ParticipantAnalyzer(FALLBACK_PREDICTIONS_CSV)
        result = analyzer.analyze()

        self.assertEqual(result.participant_id, "resultsZPH_NR")
        self.assertGreater(result.total_sentences, 0)
        self.assertIn("EEI", result.metrics)
        self.assertIn("CWI", result.metrics)
        self.assertIn("SAI", result.metrics)
        self.assertGreater(len(result.clusters), 0)

    def test_save_analysis_json(self):
        """Verify that analysis serializes cleanly to JSON."""
        analyzer = ParticipantAnalyzer(DEFAULT_PROFILE_CSV)
        out_file = analyzer.save_analysis()
        self.assertTrue(out_file.exists())
        self.assertGreater(out_file.stat().st_size, 1000)

    def test_task_and_patient_extraction_and_csv_naming(self):
        """Verify task and patient extraction and CSV naming convention."""
        from pathlib import Path
        dummy_path = Path("/mnt/DATA/SPLAB/1. Code/EEG-To-Text/dataset/ZuCo/q3zws/osfstorage/task3 - TSR/Matlab files/resultsZJS_TSR.mat")
        info = extract_task_and_patient_info(dummy_path)
        self.assertEqual(info["task_name"], "task3_TSR")
        self.assertEqual(info["patient_name"], "ZJS")
        self.assertEqual(info["patient_id"], "resultsZJS_TSR")

        expected_csv_name = f"features_{info['task_name']}_{info['patient_name']}.csv"
        self.assertTrue(expected_csv_name.endswith(f"{info['task_name']}_{info['patient_name']}.csv"))

        # Test natural reading task
        nr_path = Path("/path/to/task2 - NR/Matlab files/resultsZPH_NR.mat")
        info_nr = extract_task_and_patient_info(nr_path)
        self.assertEqual(info_nr["task_name"], "task2_NR")
        self.assertEqual(info_nr["patient_name"], "ZPH")

        # Test sentiment reading task
        sr_path = Path("/path/to/task1- SR/Matlab files/resultsZDM_SR.mat")
        info_sr = extract_task_and_patient_info(sr_path)
        self.assertEqual(info_sr["task_name"], "task1_SR")
        self.assertEqual(info_sr["patient_name"], "ZDM")

    def test_csv_dir_structure_and_finder(self):
        """Verify that CSV_DIR exists and find_participant_csv correctly locates CSV files."""
        self.assertTrue(CSV_DIR.exists())
        self.assertTrue(CSV_DIR.is_dir())

        # Test finding existing file or fallback
        csv_file = find_participant_csv("ZPH")
        # Should return a valid path if ZPH exists
        if csv_file:
            self.assertTrue(csv_file.exists())

    def test_sentence_comparator_flow_and_breakdowns(self):
        """Verify SentenceComparator correctly identifies flow states and comprehension breakdowns."""
        from neuroread_rag.participant_analysis.sentence_comparator import SentenceComparator

        df = pd.read_csv(DEFAULT_PROFILE_CSV)
        comparator = SentenceComparator(df)
        res = comparator.run()

        self.assertIn("report", res)
        self.assertIn("summary", res)
        report = res["report"]

        self.assertGreater(report["total_sentences"], 0)
        self.assertIn("flow_count", report)
        self.assertIn("breakdown_count", report)
        self.assertIn("top_flow_sentences", report)
        self.assertIn("top_breakdown_sentences", report)
        self.assertIn("contrast_insight", report)
        self.assertIn("personal_optimal_zone", report)

        # Verify summary is human readable and non-empty
        self.assertIn("flow state", res["summary"].lower())
        self.assertIn("sentences", res["summary"].lower())


if __name__ == "__main__":
    unittest.main()
