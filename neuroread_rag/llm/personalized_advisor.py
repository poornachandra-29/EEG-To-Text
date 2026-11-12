"""
Personalized Advisor — the top-level orchestrator that reads a student's
precomputed CSV, builds a structured context prompt, calls the LLM,
and returns (+ caches) a personalized learning report.

Usage:
    from neuroread_rag.llm import PersonalizedAdvisor
    advisor = PersonalizedAdvisor()
    report = advisor.generate_for_participant("resultsZPH_NR", task="task2_NR")
    print(report)

    # Or batch-generate all participants:
    advisor.batch_generate_all()
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from neuroread_rag.llm.config import LLM_CACHE_DIR
from neuroread_rag.llm.llm_client import LLMClient
from neuroread_rag.llm.prompt_builder import CognitivePromptBuilder

logger = logging.getLogger(__name__)

# Location of precomputed CSVs
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed_csvs"


class PersonalizedAdvisor:
    """End-to-end orchestrator: CSV → Profile → Prompt → LLM → Cached Report."""

    def __init__(self, backend: Optional[str] = None):
        """
        Args:
            backend: "gemini", "ollama", or "auto" (default: auto-detect)
        """
        self.client = LLMClient(backend=backend)
        self.prompt_builder = CognitivePromptBuilder()
        logger.info(
            "PersonalizedAdvisor initialized — backend: %s (%s)",
            self.client.active_backend,
            self.client.model_name,
        )

    @property
    def active_backend(self) -> str:
        return self.client.active_backend

    @property
    def model_name(self) -> str:
        return self.client.model_name

    # ------------------------------------------------------------------ #
    #  Generate for a Single Participant
    # ------------------------------------------------------------------ #
    def generate_for_participant(
        self,
        participant_id: str,
        task: str = "task2_NR",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a personalized LLM report for a single participant.

        Args:
            participant_id: e.g., "resultsZPH_NR" or "ZPH"
            task: "task1_SR", "task2_NR", or "task3_TSR"
            force_regenerate: If True, ignore cache and call LLM again

        Returns:
            Dict with keys: participant_id, task, backend, model, report_text,
                            generated_at, cached, prompt_tokens_approx
        """
        csv_path = self._find_csv(participant_id, task)
        if csv_path is None:
            return {
                "participant_id": participant_id,
                "task": task,
                "error": f"No CSV found for {participant_id} in {task}",
                "report_text": "",
            }

        # Check cache first
        cache_key = self._cache_key(participant_id, task)
        cache_path = LLM_CACHE_DIR / f"{cache_key}.json"

        if not force_regenerate and cache_path.exists():
            logger.info("Loading cached LLM report: %s", cache_path.name)
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                cached["cached"] = True
                return cached

        # Build prompt
        logger.info(
            "Generating personalized report for %s (%s) via %s...",
            participant_id, task, self.client.active_backend,
        )
        system_prompt = self.prompt_builder.get_system_prompt()
        user_prompt = self.prompt_builder.build_from_csv(csv_path)

        # Call LLM
        report_text = self.client.generate(system_prompt, user_prompt)

        # Build result
        result = {
            "participant_id": participant_id,
            "task": task,
            "backend": self.client.active_backend,
            "model": self.client.model_name,
            "report_text": report_text,
            "generated_at": datetime.now().isoformat(),
            "cached": False,
            "prompt_tokens_approx": len(user_prompt.split()) + len(system_prompt.split()),
            "csv_source": str(csv_path),
        }

        # Cache to disk
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("Cached LLM report → %s", cache_path)
        except Exception as e:
            logger.warning("Could not cache report: %s", e)

        return result

    # ------------------------------------------------------------------ #
    #  Generate from a DataFrame (in-memory, no file needed)
    # ------------------------------------------------------------------ #
    def generate_from_dataframe(
        self, df: pd.DataFrame, label: str = "custom"
    ) -> Dict[str, Any]:
        """Generate a personalized report from an in-memory DataFrame."""
        system_prompt = self.prompt_builder.get_system_prompt()
        user_prompt = self.prompt_builder.build_from_dataframe(df)

        report_text = self.client.generate(system_prompt, user_prompt)

        return {
            "participant_id": label,
            "backend": self.client.active_backend,
            "model": self.client.model_name,
            "report_text": report_text,
            "generated_at": datetime.now().isoformat(),
            "cached": False,
        }

    # ------------------------------------------------------------------ #
    #  Batch Generate All Participants
    # ------------------------------------------------------------------ #
    def batch_generate_all(
        self,
        tasks: Optional[List[str]] = None,
        force_regenerate: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized reports for ALL participants across all tasks.

        Args:
            tasks: List of tasks to process. Default: ["task2_NR"]
            force_regenerate: If True, regenerate even if cached

        Returns:
            List of result dicts, one per participant-task pair
        """
        if tasks is None:
            tasks = ["task2_NR"]

        results = []
        for task in tasks:
            task_dir = _DATA_DIR / task
            if not task_dir.exists():
                logger.warning("Task directory not found: %s", task_dir)
                continue

            csv_files = sorted(task_dir.glob("*.csv"))
            total = len(csv_files)
            print(f"\n[LLM Batch] Processing {total} participants for {task}...")

            for i, csv_path in enumerate(csv_files, 1):
                pid = csv_path.stem
                # Normalize: features_task2_NR_ZPH → resultsZPH_NR
                if pid.startswith("features_"):
                    parts = pid.replace("features_", "").split("_")
                    # parts = ["task2", "NR", "ZPH"]
                    if len(parts) >= 3:
                        subject = parts[-1]
                        pid = f"results{subject}_{task.split('_', 1)[1] if '_' in task else task}"

                print(f"  [{i}/{total}] {pid}...", end=" ", flush=True)

                try:
                    result = self.generate_for_participant(
                        pid, task=task, force_regenerate=force_regenerate
                    )
                    status = "cached" if result.get("cached") else "generated"
                    print(f"✓ ({status})")
                    results.append(result)
                except Exception as e:
                    print(f"✗ Error: {e}")
                    results.append({
                        "participant_id": pid,
                        "task": task,
                        "error": str(e),
                    })

        # Summary
        generated = sum(1 for r in results if r.get("report_text"))
        errors = sum(1 for r in results if r.get("error"))
        print(f"\n[LLM Batch] Complete: {generated} reports generated, {errors} errors")
        print(f"[LLM Batch] Reports cached in: {LLM_CACHE_DIR}")

        return results

    # ------------------------------------------------------------------ #
    #  List Available Cached Reports
    # ------------------------------------------------------------------ #
    def list_cached_reports(self) -> List[Dict[str, str]]:
        """List all cached LLM reports on disk."""
        reports = []
        for f in sorted(LLM_CACHE_DIR.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    reports.append({
                        "participant_id": data.get("participant_id", f.stem),
                        "task": data.get("task", "unknown"),
                        "backend": data.get("backend", "unknown"),
                        "model": data.get("model", "unknown"),
                        "generated_at": data.get("generated_at", "unknown"),
                        "file": str(f),
                    })
            except Exception:
                pass
        return reports

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _find_csv(self, participant_id: str, task: str) -> Optional[Path]:
        """Locate a participant's precomputed CSV file."""
        task_dir = _DATA_DIR / task
        if not task_dir.exists():
            return None

        # Try exact match first
        for csv_file in task_dir.glob("*.csv"):
            if participant_id in csv_file.stem:
                return csv_file

        # Try extracting subject code (e.g., "ZPH" from "resultsZPH_NR")
        subject = participant_id
        for prefix in ["results", "features_"]:
            subject = subject.replace(prefix, "")
        # Remove task suffix
        for suffix in ["_NR", "_TSR", "_SR"]:
            subject = subject.replace(suffix, "")

        for csv_file in task_dir.glob("*.csv"):
            if subject in csv_file.stem:
                return csv_file

        return None

    def _cache_key(self, participant_id: str, task: str) -> str:
        """Generate a stable cache key for a participant-task pair."""
        raw = f"{participant_id}_{task}_{self.client.model_name}"
        return hashlib.md5(raw.encode()).hexdigest()[:12] + f"_{participant_id}_{task}"
