"""
Lightweight REST API Server and Frontend Host for NeuroRead RAG.
Runs using Python standard library http.server, requiring zero extra server packages.
Provides REST API endpoints and serves the HTML/CSS/JS frontend dashboard.
"""

import json
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Any, Dict

# Enable running both as module and directly as python server.py
_pkg_parent = Path(__file__).resolve().parent.parent
if str(_pkg_parent) not in sys.path:
    sys.path.insert(0, str(_pkg_parent))

from neuroread_rag.config import (
    BASE_DIR,
    OUTPUT_DIR,
    CSV_DIR,
    PRECOMPUTED_CSVS_DIR,
    TASK_FOLDERS,
    METRIC_DEFINITIONS,
    DEFAULT_PROFILE_CSV,
    FALLBACK_PREDICTIONS_CSV,
    ZUCO_ROOT,
    list_zuco_tasks,
    get_active_participant_csv,
    set_active_participant_csv,
    find_participant_csv
)
from neuroread_rag.participant_analysis.mat_processor import process_mat_file
from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.grounding_engine import ResearchGroundingEngine
from neuroread_rag.reporting.report_generator import FinalReportGenerator

FRONTEND_DIR = BASE_DIR / "frontend"


class NeuroReadRequestHandler(SimpleHTTPRequestHandler):
    """Handles REST API endpoints and serves frontend static files."""

    analyzer: Optional[ParticipantAnalyzer] = None
    synthesizer: Optional[EvidenceSynthesizer] = None
    report_data: Optional[dict] = None
    participant_data: Optional[dict] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    @classmethod
    def initialize_pipeline(cls, csv_path: Optional[Path] = None):
        """Pre-compute or load participant analysis, research indexing, and full report."""
        resolved_csv = Path(csv_path).resolve() if csv_path else get_active_participant_csv()

        if cls.participant_data and cls.report_data:
            curr_pid = cls.participant_data.get("participant_id", "")
            if curr_pid and (curr_pid in resolved_csv.stem or any(p in resolved_csv.stem for p in curr_pid.split("_") if p)):
                return

        print(f"[Server] Initializing NeuroRead RAG pipeline with CSV: {resolved_csv.name}...")
        cls.analyzer = ParticipantAnalyzer(csv_path=resolved_csv)
        analysis_result = cls.analyzer.analyze()
        cls.participant_data = analysis_result.to_dict()
        pid = analysis_result.participant_id

        # Check if pre-compiled final report JSON exists on disk
        saved_report_json = OUTPUT_DIR / f"{pid}_final_report.json"
        if saved_report_json.exists():
            try:
                with open(saved_report_json, "r", encoding="utf-8") as f:
                    cls.report_data = json.load(f)
                vector_store = HybridVectorStore()
                cls.synthesizer = EvidenceSynthesizer(vector_store)
                print(f"[Server] Loaded pre-compiled report directly from disk for: {pid}")
                return
            except Exception:
                pass

        vector_store = HybridVectorStore()
        cls.synthesizer = EvidenceSynthesizer(vector_store)
        engine = ResearchGroundingEngine(cls.synthesizer)
        grounded_report = engine.ground_participant_profile(analysis_result)

        report_gen = FinalReportGenerator()
        cls.report_data = report_gen.generate_full_report(grounded_report, analysis_result)
        report_gen.save_reports(grounded_report, analysis_result)
        print(f"[Server] Pipeline ready for participant: {analysis_result.participant_id}")

    def do_GET(self):
        """Route API calls or serve static frontend assets."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Ensure pipeline is initialized
        self.initialize_pipeline()

        # API Routes
        if path == "/api/status":
            self._send_json({
                "status": "online",
                "participant_id": self.participant_data.get("participant_id"),
                "total_sentences": self.participant_data.get("total_sentences"),
                "metrics_count": len(self.participant_data.get("metrics", {}))
            })
            return

        elif path in ["/api/zuco/tasks", "/api/tasks"]:
            tasks = list_zuco_tasks()
            self._send_json({"tasks": tasks})
            return

        elif path == "/api/participants":
            available_set = set()
            for f in OUTPUT_DIR.glob("*_final_report.json"):
                available_set.add(f.stem.replace("_final_report", ""))
            if CSV_DIR.exists():
                for f in CSV_DIR.glob("*.csv"):
                    available_set.add(f.stem)
            available = sorted(list(available_set))
            self._send_json({"current": self.participant_data.get("participant_id"), "available": available})
            return

        elif path == "/api/switch_participant":
            target_pid = query_params.get("id", [""])[0]
            if target_pid:
                # Find matching features or CSV in CSV_DIR or OUTPUT_DIR
                matched_csv = find_participant_csv(target_pid)
                if not matched_csv:
                    for candidate in [OUTPUT_DIR / f"{target_pid}_features.csv", DEFAULT_PROFILE_CSV, FALLBACK_PREDICTIONS_CSV]:
                        if candidate.exists() and (target_pid in candidate.name or target_pid in candidate.stem):
                            matched_csv = candidate
                            break
                if matched_csv:
                    set_active_participant_csv(matched_csv)
                    NeuroReadRequestHandler.participant_data = None
                    NeuroReadRequestHandler.report_data = None
                    self.initialize_pipeline(csv_path=matched_csv)
                    self._send_json({"status": "switched", "participant_id": target_pid, "csv": matched_csv.name})
                    return
            self._send_json({"error": f"Participant '{target_pid}' not found"}, status=404)
            return

        elif path == "/api/participant":
            self._send_json(self.participant_data)
            return

        elif path == "/api/report":
            self._send_json(self.report_data)
            return

        elif path == "/api/metrics":
            self._send_json(METRIC_DEFINITIONS)
            return

        elif path == "/api/search":
            q = query_params.get("q", [""])[0]
            metric = query_params.get("metric", [None])[0]
            if not q and metric:
                q = METRIC_DEFINITIONS.get(metric, {}).get("name", metric)

            if not q:
                self._send_json({"error": "Query parameter 'q' or 'metric' required"}, status=400)
                return

            candidates = self.synthesizer.retriever.retrieve(q, metric_id=metric, top_k=6)
            results = []
            for chunk, score in candidates:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "paper_title": chunk.paper_title,
                    "authors": chunk.authors_short,
                    "year": chunk.year,
                    "doi": chunk.doi,
                    "section": chunk.section_name,
                    "content": chunk.content,
                    "metrics_tagged": chunk.metrics_tagged,
                    "relevance_score": round(float(score), 4),
                    "url": f"https://doi.org/{chunk.doi}" if chunk.doi else ""
                })
            self._send_json({"query": q, "metric_filter": metric, "count": len(results), "results": results})
            return

        elif path == "/api/download/report.md":
            pid = self.participant_data.get("participant_id", "participant")
            md_path = OUTPUT_DIR / f"{pid}_final_report.md"
            if md_path.exists():
                self._send_file(md_path, "text/markdown", f"{pid}_report.md")
            else:
                self.send_error(404, "Markdown report file not found")
            return

        elif path == "/api/download/report.json":
            pid = self.participant_data.get("participant_id", "participant")
            json_path = OUTPUT_DIR / f"{pid}_final_report.json"
            if json_path.exists():
                self._send_file(json_path, "application/json", f"{pid}_report.json")
            else:
                self.send_error(404, "JSON report file not found")
            return

        # Fallback to serving static HTML/CSS/JS files
        super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        """Handle execution requests from dashboard (e.g. running pipeline on selected task/patient)."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/pipeline/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            task_name = data.get("task")
            patient_file = data.get("patient_mat") or data.get("patient") or data.get("filename")
            mat_path_str = data.get("mat_path")
            csv_path_str = data.get("csv_path")

            # 1. First, search for an existing precomputed CSV
            target_csv: Optional[Path] = None
            if csv_path_str and Path(csv_path_str).exists():
                target_csv = Path(csv_path_str).resolve()
            elif patient_file:
                clean_name = patient_file.replace(".mat", "").replace(".csv", "").strip()
                if task_name:
                    norm_task = task_name.replace(" ", "").replace("-", "_").strip("_")
                    t_dir = TASK_FOLDERS.get(norm_task) or (PRECOMPUTED_CSVS_DIR / norm_task)
                    if t_dir.exists():
                        for cand in [
                            t_dir / f"features_{norm_task}_{clean_name}.csv",
                            t_dir / f"features_{norm_task}_{clean_name.replace('results', '')}.csv",
                            t_dir / f"{clean_name}.csv",
                            t_dir / f"{patient_file}.csv"
                        ]:
                            if cand.exists():
                                target_csv = cand
                                break
                if not target_csv:
                    target_csv = find_participant_csv(clean_name)

            # 2. Second, search for raw .mat file if precomputed CSV is not found
            target_mat: Optional[Path] = None
            if not target_csv:
                if mat_path_str and Path(mat_path_str).exists():
                    target_mat = Path(mat_path_str).resolve()
                elif task_name and patient_file and ZUCO_ROOT.exists():
                    candidate = ZUCO_ROOT / task_name / "Matlab files" / patient_file
                    if candidate.exists():
                        target_mat = candidate.resolve()
                elif patient_file and ZUCO_ROOT.exists():
                    for candidate in ZUCO_ROOT.glob(f"task*/Matlab files/{patient_file}"):
                        if candidate.exists():
                            target_mat = candidate.resolve()
                            break

            if not target_csv and not target_mat:
                self._send_json({
                    "error": f"Neither precomputed CSV nor MAT file found for task='{task_name}', patient='{patient_file}'"
                }, status=404)
                return

            try:
                if target_csv:
                    print(f"\n[Server] Using precomputed CSV: {target_csv.name} ({target_csv.parent.name}) - Instant Execution!")
                    csv_path = target_csv
                    set_active_participant_csv(csv_path)
                else:
                    print(f"\n[Server] Computing from raw MAT file: {target_mat.name} ({target_mat.parent.parent.name})...")
                    csv_path = process_mat_file(target_mat)
                    set_active_participant_csv(csv_path)

                # Check if pre-compiled report JSON already exists for instant serving
                participant_stem = target_csv.stem if target_csv else target_mat.stem
                clean_pid = participant_stem.replace("features_", "")
                cached_report_file = OUTPUT_DIR / f"{clean_pid}_final_report.json"
                cached_analysis_file = OUTPUT_DIR / f"{clean_pid}_analysis.json"

                if cached_report_file.exists() and cached_analysis_file.exists():
                    print(f"[Server] Loading cached pre-compiled analysis and report for: {clean_pid} (0.01s)")
                    report_data = json.loads(cached_report_file.read_text(encoding="utf-8"))
                    participant_data = json.loads(cached_analysis_file.read_text(encoding="utf-8"))
                    NeuroReadRequestHandler.report_data = report_data
                    NeuroReadRequestHandler.participant_data = participant_data
                    self._send_json({
                        "status": "success",
                        "participant_id": clean_pid,
                        "sentences_count": participant_data.get("total_sentences", 0),
                        "task": task_name or (target_csv.parent.name if target_csv else target_mat.parent.parent.name),
                        "filename": (target_csv.name if target_csv else target_mat.name),
                        "report": report_data
                    })
                    return

                # Otherwise, execute deterministic Participant Analysis & Grounding
                analyzer = ParticipantAnalyzer(csv_path=csv_path)
                analysis_result = analyzer.analyze()
                analyzer.save_analysis()

                vector_store = HybridVectorStore()
                synthesizer = EvidenceSynthesizer(vector_store)
                engine = ResearchGroundingEngine(synthesizer)
                grounded_report = engine.ground_participant_profile(analysis_result)
                engine.save_grounded_report(grounded_report)

                report_gen = FinalReportGenerator()
                report_data = report_gen.generate_full_report(grounded_report, analysis_result)
                report_gen.save_reports(grounded_report, analysis_result)

                NeuroReadRequestHandler.analyzer = analyzer
                NeuroReadRequestHandler.synthesizer = synthesizer
                NeuroReadRequestHandler.participant_data = analysis_result.to_dict()
                NeuroReadRequestHandler.report_data = report_data

                print(f"[Server] Pipeline run completed successfully for: {analysis_result.participant_id} ({analysis_result.total_sentences} sentences).")
                self._send_json({
                    "status": "success",
                    "participant_id": analysis_result.participant_id,
                    "sentences_count": analysis_result.total_sentences,
                    "task": task_name or (target_csv.parent.name if target_csv else target_mat.parent.parent.name),
                    "filename": (target_csv.name if target_csv else target_mat.name),
                    "report": report_data
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": f"Pipeline execution failed: {str(e)}"}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str, filename: str):
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int = 8080, host: str = "127.0.0.1", csv_path: Optional[Path] = None):
    """Launch the NeuroRead RAG server."""
    NeuroReadRequestHandler.initialize_pipeline(csv_path=csv_path)
    server_address = (host, port)
    httpd = HTTPServer(server_address, NeuroReadRequestHandler)
    print(f"\n=======================================================")
    print(f" NeuroRead AI RAG Dashboard Server Running!")
    print(f" Dashboard URL: http://{host}:{port}/")
    print(f" API Endpoints: http://{host}:{port}/api/report")
    print(f" Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
