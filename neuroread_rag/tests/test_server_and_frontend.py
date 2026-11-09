"""
Tests for Phase 6: REST API Server and Frontend Integration.
"""

import unittest
from pathlib import Path
from neuroread_rag.config import BASE_DIR, OUTPUT_DIR
from neuroread_rag.server import NeuroReadRequestHandler, FRONTEND_DIR


class TestServerAndFrontend(unittest.TestCase):
    """Test suite for Phase 6 REST API endpoints and static assets."""

    @classmethod
    def setUpClass(cls):
        NeuroReadRequestHandler.initialize_pipeline()

    def test_frontend_static_files_exist(self):
        """Verify all HTML, CSS, and JS files exist in neuroread_rag/frontend."""
        self.assertTrue((FRONTEND_DIR / "index.html").exists())
        self.assertTrue((FRONTEND_DIR / "style.css").exists())
        self.assertTrue((FRONTEND_DIR / "app.js").exists())

        # Check HTML content contains all required dashboard sections
        html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("Executive Overview", html_content)
        self.assertIn("Focus & Attention Report", html_content)
        self.assertIn("Cognitive Workload Report", html_content)
        self.assertIn("Metric Scientific Assessment", html_content)
        self.assertIn("Research Evidence Comparison", html_content)
        self.assertIn("Research Explorer", html_content)
        self.assertIn("Personalized Recommendations", html_content)
        self.assertIn("Complete AI Assessment Report", html_content)

    def test_pipeline_initialized_on_server(self):
        """Verify pipeline objects are properly prepared on server."""
        self.assertIsNotNone(NeuroReadRequestHandler.participant_data)
        self.assertIsNotNone(NeuroReadRequestHandler.report_data)
        self.assertEqual(NeuroReadRequestHandler.participant_data["participant_id"], "resultsZPH_NR")

    def test_search_retrieval_api(self):
        """Verify server search retrieval handles queries and returns paper chunks."""
        synthesizer = NeuroReadRequestHandler.synthesizer
        self.assertIsNotNone(synthesizer)

        candidates = synthesizer.retriever.retrieve("frontal midline theta", metric_id="FMT", top_k=3)
        self.assertGreater(len(candidates), 0)
        chunk, score = candidates[0]
        self.assertTrue(len(chunk.paper_title) > 0)
        self.assertTrue(len(chunk.doi) > 0)


if __name__ == "__main__":
    unittest.main()
