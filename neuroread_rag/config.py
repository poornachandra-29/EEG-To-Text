"""
Configuration and constants for the NeuroRead RAG module.
Ensures strict separation between physiological observations and clinical/psychological claims.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

# Workspace Root & Default Directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = OUTPUT_DIR / "csv_features"
CSV_DIR.mkdir(parents=True, exist_ok=True)
ZUCO_ROOT = PROJECT_ROOT / "dataset" / "ZuCo" / "q3zws" / "osfstorage"

# Input CSV Paths from Existing Pipeline
DEFAULT_PROFILE_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "cognitive_profile_output.csv"
FALLBACK_PREDICTIONS_CSV = PROJECT_ROOT / "test_nrei_csv" / "test_predictions.csv"
PERSONAL_RANGES_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "personal_reference_ranges.csv"
CLUSTER_INTERPRETATION_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "cluster_cognitive_interpretation.csv"
LATEST_PARTICIPANT_FILE = OUTPUT_DIR / "latest_participant.txt"


def list_zuco_tasks() -> List[Dict[str, Any]]:
    """Scan ZuCo directory and return available study reading tasks and student .mat files."""
    task_order = ["task3 - TSR", "task2 - NR", "task1- SR"]
    task_labels = {
        "task3 - TSR": "Study Material 3: Task-Specific Reading (TSR)",
        "task2 - NR": "Study Material 2: Natural Reading (NR)",
        "task1- SR": "Study Material 1: Semantic Reading (SR)"
    }
    tasks = []
    if not ZUCO_ROOT.exists():
        return tasks

    def sort_key(d: Path) -> int:
        return task_order.index(d.name) if d.name in task_order else 99

    for t_dir in sorted(ZUCO_ROOT.glob("task*"), key=sort_key):
        mat_folder = t_dir / "Matlab files"
        if not mat_folder.is_dir():
            continue
        mat_files = sorted(mat_folder.glob("*.mat"), key=lambda p: p.name.lower())
        patients = []
        for mf in mat_files:
            stem = mf.stem  # e.g. resultsZJS_TSR
            code = stem.replace("results", "").split("_")[0]
            patients.append({
                "filename": mf.name,
                "patient_id": stem,
                "student_id": stem,
                "patient_code": code,
                "student_code": code,
                "full_path": str(mf.resolve())
            })
        tasks.append({
            "task_id": t_dir.name,
            "task_name": task_labels.get(t_dir.name, t_dir.name),
            "mat_folder": str(mat_folder.resolve()),
            "patient_count": len(patients),
            "student_count": len(patients),
            "patients": patients
        })
    return tasks


def set_active_participant_csv(csv_path: Path) -> None:
    """Save the path to the most recently analyzed participant CSV."""
    try:
        LATEST_PARTICIPANT_FILE.write_text(str(Path(csv_path).resolve()), encoding="utf-8")
    except Exception:
        pass


def get_active_participant_csv() -> Path:
    """
    Get the active participant CSV:
    1. Returns latest recorded in latest_participant.txt if valid.
    2. Otherwise, returns the most recently modified *.csv in CSV_DIR (separate folder).
    3. Otherwise, returns the most recently modified *_features.csv in OUTPUT_DIR.
    4. Otherwise, falls back to DEFAULT_PROFILE_CSV or FALLBACK_PREDICTIONS_CSV.
    """
    if LATEST_PARTICIPANT_FILE.exists():
        try:
            saved_str = LATEST_PARTICIPANT_FILE.read_text(encoding="utf-8").strip()
            saved_path = Path(saved_str)
            if saved_path.exists():
                return saved_path
        except Exception:
            pass

    # Check for recent CSVs in dedicated CSV_DIR
    if CSV_DIR.exists():
        recent_csvs = sorted(
            CSV_DIR.glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if recent_csvs:
            return recent_csvs[0]

    # Check for recent *_features.csv in OUTPUT_DIR
    recent_features = sorted(
        OUTPUT_DIR.glob("*_features.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if recent_features:
        return recent_features[0]

    if DEFAULT_PROFILE_CSV.exists():
        return DEFAULT_PROFILE_CSV
    if FALLBACK_PREDICTIONS_CSV.exists():
        return FALLBACK_PREDICTIONS_CSV

    raise FileNotFoundError("No participant CSV or features file found.")


def find_participant_csv(identifier: str) -> Optional[Path]:
    """
    Find existing CSV for a participant by ID, code, or task name in CSV_DIR or OUTPUT_DIR.
    Matches e.g. 'ZJS', 'resultsZJS_TSR', 'task3_TSR_ZJS', etc.
    """
    if not identifier:
        return None
    clean_id = identifier.replace(".mat", "").replace(".csv", "").strip().lower()
    
    # 1. Search in dedicated CSV_DIR
    if CSV_DIR.exists():
        for f in sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            f_stem_lower = f.stem.lower()
            if clean_id in f_stem_lower:
                return f
                
    # 2. Search in OUTPUT_DIR
    for f in sorted(OUTPUT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        f_stem_lower = f.stem.lower()
        if clean_id in f_stem_lower:
            return f
            
    return None

# Metric Definitions and Metadata
METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    # EEG Spectral & Derived Metrics
    "EEI": {
        "name": "EEG Engagement Index",
        "category": "EEG",
        "unit": "ratio (Beta / (Alpha + Theta))",
        "description": "Calculated as Beta power divided by (Alpha + Theta) power. Reflects relative spectral engagement during reading tasks.",
        "search_keywords": ["EEG engagement index", "beta alpha theta ratio", "cognitive engagement EEG reading"],
    },
    "FMT": {
        "name": "Frontal Midline Theta",
        "category": "EEG",
        "unit": "µV²",
        "description": "Spectral power in the theta band (4-8 Hz) localized across frontal midline electrodes. Often associated with working memory and mental effort.",
        "search_keywords": ["frontal midline theta", "FMT cognitive workload", "theta oscillations working memory"],
    },
    "ASM": {
        "name": "Alpha Asymmetry Metric",
        "category": "EEG",
        "unit": "log difference",
        "description": "Hemispheric frontal asymmetry between right and left alpha power. Explores approach vs withdrawal related affective engagement.",
        "search_keywords": ["frontal alpha asymmetry", "EEG alpha asymmetry reading", "approach withdrawal motivation EEG"],
    },
    "ThetaPower": {
        "name": "Theta Band Power",
        "category": "EEG",
        "unit": "µV²",
        "description": "Total power in the 4-8 Hz frequency band across recorded scalp channels.",
        "search_keywords": ["EEG theta power", "theta oscillations reading", "cognitive effort theta band"],
    },
    "AlphaPower": {
        "name": "Alpha Band Power",
        "category": "EEG",
        "unit": "µV²",
        "description": "Total power in the 8-12 Hz frequency band. Linked to cortical gating, relaxation, and attentional focus.",
        "search_keywords": ["EEG alpha suppression", "alpha power reading comprehension", "cortical idling alpha band"],
    },
    "BetaPower": {
        "name": "Beta Band Power",
        "category": "EEG",
        "unit": "µV²",
        "description": "Total power in the 13-30 Hz frequency band. Linked to active concentration and cognitive task execution.",
        "search_keywords": ["EEG beta band", "beta power attention", "sensorimotor beta reading"],
    },
    "Theta_Alpha_Ratio": {
        "name": "Theta / Alpha Ratio",
        "category": "EEG",
        "unit": "ratio",
        "description": "Ratio of theta power to alpha power. Informative regarding drowsiness, fatigue-compatible patterns, or shifts in alertness.",
        "search_keywords": ["theta alpha ratio EEG", "EEG drowsiness alertness ratio", "fatigue theta alpha oscillations"],
    },
    "Theta_Beta_Ratio": {
        "name": "Theta / Beta Ratio",
        "category": "EEG",
        "unit": "ratio",
        "description": "Ratio of theta power to beta power. Widely studied in cognitive control and vigilance maintenance literature.",
        "search_keywords": ["theta beta ratio", "TBR cognitive control", "executive attention theta beta ratio"],
    },

    # Eye-Tracking Metrics
    "GD": {
        "name": "Gaze Duration",
        "category": "Eye-Tracking",
        "unit": "ms",
        "description": "Sum of all fixations on a word before the eyes leave it. Indicates early lexical processing and initial syntactic difficulty.",
        "search_keywords": ["gaze duration eye tracking", "gaze duration reading comprehension", "eye movements word processing"],
    },
    "FFD": {
        "name": "First Fixation Duration",
        "category": "Eye-Tracking",
        "unit": "ms",
        "description": "Duration of the very first fixation on a word during initial forward reading pass.",
        "search_keywords": ["first fixation duration reading", "FFD eye tracking psycholinguistics", "early lexical access fixation duration"],
    },
    "TRT": {
        "name": "Total Reading Time",
        "category": "Eye-Tracking",
        "unit": "ms",
        "description": "Total duration of all fixations on a word or sentence including regressions and rereading passes.",
        "search_keywords": ["total reading time eye tracking", "total dwell time reading difficulty", "rereading regression duration"],
    },
    "PupilSize": {
        "name": "Mean Pupil Diameter",
        "category": "Eye-Tracking",
        "unit": "arbitrary units / px",
        "description": "Averaged pupillometric dilation across fixation samples. Index of autonomic arousal, locus coeruleus norepinephrine system activity, and cognitive load.",
        "search_keywords": ["pupil size cognitive workload", "pupillometry reading effort", "locus coeruleus pupil dilation task difficulty"],
    },
    "FixationCount": {
        "name": "Fixation Count",
        "category": "Eye-Tracking",
        "unit": "count",
        "description": "Average number of distinct fixations per word/sentence. Correlates with sentence complexity and processing demand.",
        "search_keywords": ["fixation count reading comprehension", "fixation frequency eye tracking text complexity"],
    },

    # Composite Cognitive/Behavioral Proxies
    "REI": {
        "name": "Engagement-Related Signal Index",
        "category": "Composite Proxy",
        "unit": "standardized robust-z",
        "description": "Composite heuristic combining EEI, FMT, and PupilSize. Reflects concurrent neurophysiological activation.",
        "search_keywords": ["multimodal cognitive engagement index", "EEG pupillometry engagement fusion"],
    },
    "CWI": {
        "name": "Cognitive Workload Evidence Index",
        "category": "Composite Proxy",
        "unit": "standardized robust-z",
        "description": "Composite heuristic combining theta power, pupil dilation, gaze duration, and total reading time.",
        "search_keywords": ["multimodal cognitive workload index", "eye tracking EEG mental effort fusion"],
    },
    "SAI": {
        "name": "Engagement-Stability Evidence Proxy",
        "category": "Composite Proxy",
        "unit": "standardized robust-z",
        "description": "REI penalized by rolling local signal and reading variability. Measures consistent sustained engagement pattern.",
        "search_keywords": ["sustained attention variability index", "intraindividual variability cognitive performance"],
    },
    "MWLI_proxy": {
        "name": "Attention-Instability Evidence Proxy",
        "category": "Composite Proxy",
        "unit": "standardized robust-z",
        "description": "Inverse EEI combined with eye-tracking reading time variability. Indicates periods of unstable processing or erratic reading cadence.",
        "search_keywords": ["mind wandering eye tracking reading", "attention lapse EEG reading variability"],
    },
    "VSI_proxy": {
        "name": "Signal-Stability Proxy",
        "category": "Composite Proxy",
        "unit": "standardized robust-z",
        "description": "Inverse of total rolling neurophysiological variability across reading intervals.",
        "search_keywords": ["signal stability cognitive processing", "physiological signal variance reading task"],
    },
}

# Scientific Guardrails: Mandatory Non-Diagnostic Disclaimers
MANDATORY_DISCLAIMERS = [
    "This system provides research-grounded physiological observations, NOT clinical diagnoses.",
    "Composite indices (SAI, REI, MWLI_proxy, CWI, VSI_proxy) are exploratory engineering proxies, not validated psychometric or clinical scales.",
    "Measurements reflect within-session reading patterns relative to the participant's own baseline, not normative population abnormalities.",
    "Variations in gaze, pupil dilation, or spectral EEG power do not diagnose ADHD, executive dysfunction, or clinical reading disorders.",
    "Attention instability indicators reflect cadence irregularities and spectral shifts, not definitive subjective mind-wandering states."
]
