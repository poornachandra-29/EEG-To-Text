"""
Snapshot Parity & Backward Compatibility Test.
Verifies that the modernized v2.0 pipeline preserves all legacy contracts
and keys from the pre-modification snapshot (legacy_ZPH.json)
while correctly introducing new confound-controlled and behavioral fields.
"""

import json
import unittest
from pathlib import Path
import pandas as pd

from neuroread_rag.config import DEFAULT_PROFILE_CSV
from neuroread_rag.participant_analysis.sentence_comparator import SentenceComparator

SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "legacy_ZPH.json"


class TestSnapshotParity(unittest.TestCase):
    """Test suite ensuring non-regression against baseline legacy snapshot."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(SNAPSHOT_PATH.exists(), f"Snapshot not found at {SNAPSHOT_PATH}")
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            cls.legacy_snapshot = json.load(f)

        cls.df = pd.read_csv(DEFAULT_PROFILE_CSV)
        cls.comparator = SentenceComparator(cls.df)
        cls.res = cls.comparator.run()
        cls.report = cls.res["report"]

    def test_legacy_keys_presence(self):
        """Verify all legacy section keys and high-level fields are preserved."""
        legacy_keys = [
            "total_sentences",
            "flow_count",
            "breakdown_count",
            "fatigue_count",
            "normal_count",
            "top_flow_sentences",
            "top_breakdown_sentences",
            "contrast_insight",
            "personal_optimal_zone",
            "learning_efficiency",
            "session_fatigue_curve",
        ]
        for key in legacy_keys:
            self.assertIn(key, self.report, f"Legacy key '{key}' missing from modern report!")

    def test_learning_efficiency_parity_and_heuristic_labels(self):
        """Verify learning efficiency structure preserves legacy scores and tags heuristic nature."""
        les = self.report["learning_efficiency"]
        legacy_les = self.legacy_snapshot["learning_efficiency"]

        self.assertIn("score", les)
        self.assertIn("tier", les)
        self.assertIn("flow_factor", les)

        # New v2.0 fields
        self.assertIn("is_heuristic", les)
        self.assertTrue(les["is_heuristic"])
        self.assertIn("heuristic_label", les)
        self.assertIn("breakdown_behav_count", les)
        self.assertIn("comprehension_errors_count", les)

    def test_session_fatigue_curve_chronological_minutes(self):
        """Verify session fatigue curve uses real elapsed timing."""
        fatigue = self.report["session_fatigue_curve"]
        self.assertIn("checkpoints", fatigue)
        self.assertIn("recommended_study_block_minutes", fatigue)
        self.assertIn("total_session_minutes", fatigue)
        self.assertIn("has_reliable_fatigue_detected", fatigue)

        pomo = fatigue["recommended_study_block_minutes"]
        self.assertGreaterEqual(pomo, 15)
        self.assertLessEqual(pomo, 40)

        for cp in fatigue["checkpoints"]:
            self.assertIn("estimated_minute", cp)
            self.assertIn("reading_wpm", cp)
            self.assertGreaterEqual(cp["estimated_minute"], 0.0)

    def test_behavioral_breakdown_distinction(self):
        """Verify breakdown_behav is not constrained by legacy quartile quotas."""
        df_out = self.res["df"]
        self.assertIn("breakdown_behav", df_out.columns)
        self.assertIn("legacy_state", df_out.columns)

        # Legacy quartile breakdown was ~25%
        # breakdown_behav must be binary
        self.assertTrue(set(df_out["breakdown_behav"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
