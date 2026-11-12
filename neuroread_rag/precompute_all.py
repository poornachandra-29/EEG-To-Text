"""
Batch Precomputation Script for NeuroRead RAG.
Precomputes all patient CSVs and pre-compiles analysis reports across all tasks:
  - Task 1: Semantic Reading (SR) (5 patients)
  - Task 2: Normal Reading (NR) (12 patients)
  - Task 3: Task-Specific Reading (TSR) (12 patients)

All CSVs are saved in self-contained task folders inside neuroread_rag/data/precomputed_csvs/:
  - neuroread_rag/data/precomputed_csvs/task1_SR/features_task1_SR_<patient>.csv
  - neuroread_rag/data/precomputed_csvs/task2_NR/features_task2_NR_<patient>.csv
  - neuroread_rag/data/precomputed_csvs/task3_TSR/features_task3_TSR_<patient>.csv

Usage:
    python -m neuroread_rag.precompute_all
    python -m neuroread_rag.precompute_all --force
    python -m neuroread_rag.precompute_all --task task2_NR
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Enable running both as module and directly as python precompute_all.py
_pkg_parent = Path(__file__).resolve().parent.parent
if str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

from neuroread_rag.config import (
    BASE_DIR,
    ZUCO_ROOT,
    PRECOMPUTED_CSVS_DIR,
    TASK_FOLDERS,
    set_active_participant_csv
)
from neuroread_rag.participant_analysis.mat_processor import process_mat_file, extract_task_and_patient_info
from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine
from neuroread_rag.reporting.report_generator import FinalReportGenerator


def discover_all_mat_files(task_filter: str = None) -> List[Dict[str, Any]]:
    """Scan ZuCo directory and return list of all .mat files with task metadata."""
    if not ZUCO_ROOT.exists():
        raise FileNotFoundError(f"ZuCo raw directory not found: {ZUCO_ROOT}")

    task_dir_aliases = {
        "task3 - TSR": "task3_TSR",
        "task2 - NR": "task2_NR",
        "task1- SR": "task1_SR",
        "task1 - SR": "task1_SR"
    }

    discovered = []
    for t_dir in sorted(ZUCO_ROOT.glob("task*")):
        norm_task = task_dir_aliases.get(t_dir.name, t_dir.name.replace(" ", "").replace("-", "_").strip("_"))
        if task_filter and task_filter.lower() not in norm_task.lower():
            continue

        mat_dir = t_dir / "Matlab files"
        if not mat_dir.is_dir():
            continue

        for mf in sorted(mat_dir.glob("*.mat"), key=lambda p: p.name.lower()):
            info = extract_task_and_patient_info(mf)
            # Use normalized task folder name
            info["task_name"] = norm_task
            info["mat_path"] = mf
            discovered.append(info)

    return discovered


def precompute_participant(
    item: Dict[str, Any],
    force: bool = False,
    generate_reports: bool = True,
    engine: ResearchGroundingEngine = None,
    report_gen: FinalReportGenerator = None
) -> Dict[str, Any]:
    """Process a single participant .mat file and save CSV to task folder."""
    task_name = item["task_name"]
    patient_name = item["patient_name"]
    patient_id = item["patient_id"]
    mat_path = item["mat_path"]

    task_folder = TASK_FOLDERS.get(task_name, PRECOMPUTED_CSVS_DIR / task_name)
    task_folder.mkdir(parents=True, exist_ok=True)
    expected_csv = task_folder / f"features_{task_name}_{patient_name}.csv"

    t0 = time.time()
    csv_existed = expected_csv.exists()

    if csv_existed and not force:
        csv_path = expected_csv
        status = "SKIPPED (Already exists)"
    else:
        try:
            csv_path = process_mat_file(mat_path)
            status = "COMPUTED"
        except Exception as exc:
            print(f"[Precompute Error] Failed processing {mat_path.name}: {exc}")
            return {
                "task_name": task_name,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "csv_path": None,
                "status": f"FAILED ({exc})",
                "report_status": "SKIPPED",
                "elapsed_sec": round(time.time() - t0, 2)
            }

    elapsed = time.time() - t0

    # Optional: precompute grounding report
    report_status = "N/A"
    if generate_reports and engine and report_gen:
        try:
            analyzer = ParticipantAnalyzer(csv_path)
            analysis_result = analyzer.analyze()
            analyzer.save_analysis()

            grounded_report = engine.ground_participant_profile(analysis_result)
            engine.save_grounded_report(grounded_report)
            report_gen.save_reports(grounded_report, analysis_result)
            report_status = "REPORTS_GENERATED"
        except Exception as e:
            report_status = f"REPORT_ERR ({e})"

    return {
        "task_name": task_name,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "csv_path": str(csv_path),
        "status": status,
        "report_status": report_status,
        "elapsed_sec": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute all patient CSVs for NeuroRead RAG")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing CSVs")
    parser.add_argument("--task", type=str, default=None, help="Filter to specific task (e.g. task1_SR, task2_NR, task3_TSR)")
    parser.add_argument("--skip-reports", action="store_true", help="Skip pre-generating full reports (only generate CSVs)")
    args = parser.parse_args()

    print("=" * 72)
    print(" 🚀 NeuroRead RAG: Patient CSV Batch Precomputer")
    print(f" Target Folder: {PRECOMPUTED_CSVS_DIR}")
    print("=" * 72)

    mat_items = discover_all_mat_files(task_filter=args.task)
    print(f"\nDiscovered {len(mat_items)} total MATLAB participant files across tasks:")
    for item in mat_items:
        print(f"  • {item['task_name']:<12} | {item['patient_id']:<20} ({item['mat_path'].name})")

    # Initialize shared grounding engine & report generator if generating reports
    engine = None
    report_gen = None
    if not args.skip_reports:
        print("\n[Precompute] Initializing RAG Hybrid Vector Store and Grounding Engine...")
        vs = HybridVectorStore()
        synthesizer = EvidenceSynthesizer(vs)
        engine = ResearchGroundingEngine(synthesizer)
        report_gen = FinalReportGenerator()

    print("\n[Precompute] Starting batch processing...")
    results = []
    for idx, item in enumerate(mat_items, 1):
        print(f"\n[{idx:2d}/{len(mat_items)}] Processing {item['task_name']} / {item['patient_id']}...")
        res = precompute_participant(
            item,
            force=args.force,
            generate_reports=not args.skip_reports,
            engine=engine,
            report_gen=report_gen
        )
        results.append(res)
        csv_name = Path(res["csv_path"]).name if res.get("csv_path") else "FAILED"
        print(f"      -> CSV: {csv_name} [{res['status']}] ({res['elapsed_sec']}s)")
        if not args.skip_reports:
            print(f"      -> Reports: {res['report_status']}")

    # Set last processed as active
    if results:
        set_active_participant_csv(Path(results[-1]["csv_path"]))

    print("\n" + "=" * 72)
    print(" 🎉 Precomputation Complete!")
    print(f" Total participants processed: {len(results)}")
    print(f" Task Folders created in {PRECOMPUTED_CSVS_DIR}:")
    for t_key, t_dir in TASK_FOLDERS.items():
        count = len(list(t_dir.glob("*.csv")))
        print(f"   • {t_key}: {count} CSV files in {t_dir.name}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
