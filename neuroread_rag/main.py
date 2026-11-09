"""
NeuroRead RAG Main CLI Entrypoint.
Execute participant grounding, report generation, or launch the interactive dashboard server.

Usage:
    # 1. Run full analysis and generate all reports (Markdown, HTML, JSON):
    python -m neuroread_rag.main --generate-report

    # 2. Launch the interactive web dashboard on localhost:
    python -m neuroread_rag.main --server --port 8080
"""

import argparse
import sys
from pathlib import Path

from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.participant_analysis.mat_processor import process_mat_file
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.config import (
    get_active_participant_csv,
    set_active_participant_csv,
    find_participant_csv,
    list_zuco_tasks,
    ZUCO_ROOT,
    OUTPUT_DIR,
    CSV_DIR
)
from neuroread_rag.ingestion.pipeline import IngestionPipeline
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine
from neuroread_rag.reporting.report_generator import FinalReportGenerator
from neuroread_rag.server import run_server


def interactive_select_patient() -> Path:
    """Terminal interactive menu to select a ZuCo task folder and patient .mat file."""
    tasks = list_zuco_tasks()
    if not tasks:
        raise FileNotFoundError(f"No ZuCo tasks found at: {ZUCO_ROOT}")

    print("\n" + "=" * 64)
    print(" 🧠 NeuroRead AI: Interactive ZuCo Task & Patient Selector")
    print("=" * 64)
    print("\nSelect ZuCo Task:")
    for idx, t in enumerate(tasks, 1):
        print(f"  [{idx}] {t['task_name']} ({t['patient_count']} patients)")

    task_idx = 0
    try:
        choice = input(f"\nEnter task number (1-{len(tasks)}) [default: 1]: ").strip()
        if choice and choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(tasks):
                task_idx = val - 1
    except (EOFError, KeyboardInterrupt):
        pass

    selected_task = tasks[task_idx]
    patients = selected_task["patients"]

    print(f"\nSelect Patient in '{selected_task['task_name']}':")
    for idx, p in enumerate(patients, 1):
        print(f"  [{idx:2d}] {p['filename']:<22} (Patient {p['patient_code']})")

    patient_idx = 0
    try:
        choice = input(f"\nEnter patient number (1-{len(patients)}) [default: 1]: ").strip()
        if choice and choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(patients):
                patient_idx = val - 1
    except (EOFError, KeyboardInterrupt):
        pass

    selected_patient = patients[patient_idx]
    print(f"\n✓ Selected: {selected_patient['filename']} ({selected_task['task_name']})")
    return Path(selected_patient["full_path"])


def main():
    parser = argparse.ArgumentParser(description="NeuroRead Research-Grounded RAG Pipeline")
    parser.add_argument("--interactive", "--select", action="store_true", help="Launch interactive terminal selector for task and patient")
    parser.add_argument("--csv", type=str, default=None, help="Path to input participant CSV (default: latest active participant)")
    parser.add_argument("--mat", type=str, default=None, help="Path to input participant MATLAB .mat file (e.g. resultsZJS_TSR.mat)")
    parser.add_argument("--patient", "--participant", type=str, default=None, help="Select participant ID (e.g. resultsZJS_TSR or resultsZPH_NR)")
    parser.add_argument("--generate-report", action="store_true", help="Generate 10-section report in MD, HTML, and JSON")
    parser.add_argument("--server", action="store_true", help="Launch interactive web dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Port for dashboard server (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for dashboard server (default: 127.0.0.1)")

    args = parser.parse_args()

    # If user ran `python -m neuroread_rag.main` with no args or --interactive:
    is_bare_run = not any([args.mat, args.csv, args.patient, args.server, args.generate_report])
    if args.interactive or is_bare_run:
        target_mat = interactive_select_patient()
        args.mat = str(target_mat)
        args.generate_report = True

    # Resolve CSV: either extracted from .mat, specified by --csv, specified by --patient, or latest active
    csv_path = None
    if args.mat:
        print(f"\n[NeuroRead RAG] 0. Extracting features directly from MATLAB file: {args.mat}")
        csv_path = process_mat_file(args.mat)
        set_active_participant_csv(csv_path)
    elif args.patient:
        candidate = find_participant_csv(args.patient)
        if candidate and candidate.exists():
            csv_path = candidate
        else:
            csv_path = get_active_participant_csv()
        set_active_participant_csv(csv_path)
    elif args.csv:
        csv_path = Path(args.csv).resolve()
        set_active_participant_csv(csv_path)
    else:
        csv_path = get_active_participant_csv()

    # If user requested --server without --generate-report, start server immediately without re-calculating
    if args.server and not args.generate_report:
        run_server(port=args.port, host=args.host, csv_path=csv_path)
        return

    # 1. Pipeline execution
    print("\n[NeuroRead RAG] 1. Analyzing participant data...")
    analyzer = ParticipantAnalyzer(csv_path)
    analysis_result = analyzer.analyze()
    analyzer.save_analysis()
    print(f"  ✓ Processed participant: {analysis_result.participant_id} ({analysis_result.total_sentences} sentences, {len(analysis_result.metrics)} metrics)")

    print("[NeuroRead RAG] 2. Ingesting scientific literature...")
    pipeline = IngestionPipeline()
    stats = pipeline.run_ingestion()
    print(f"  ✓ Indexed {stats['papers_count']} papers, {stats['chunks_count']} scientific chunks, {stats['evidence_count']} claims")

    print("[NeuroRead RAG] 3. Synthesizing multi-paper evidence & grounding measurements...")
    vector_store = HybridVectorStore()
    synthesizer = EvidenceSynthesizer(vector_store)
    engine = ResearchGroundingEngine(synthesizer)
    grounded_report = engine.ground_participant_profile(analysis_result)
    engine.save_grounded_report(grounded_report)
    print("  ✓ Research grounding complete.")

    print("[NeuroRead RAG] 4. Compiling 10-section assessment report...")
    report_gen = FinalReportGenerator()
    out_files = report_gen.save_reports(grounded_report, analysis_result)
    print(f"  ✓ Markdown Report: {out_files['markdown']}")
    print(f"  ✓ HTML Report:     {out_files['html']}")
    print(f"  ✓ JSON Report:     {out_files['json']}")

    # If bare run, ask to launch server
    if is_bare_run and not args.server:
        try:
            launch_ui = input("\nLaunch interactive web dashboard now? [Y/n]: ").strip().lower()
            if launch_ui in ["", "y", "yes"]:
                args.server = True
        except (EOFError, KeyboardInterrupt):
            pass

    if args.server:
        run_server(port=args.port, host=args.host, csv_path=csv_path)


if __name__ == "__main__":
    main()
