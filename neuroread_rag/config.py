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

# Self-contained Data and Artifact Directories
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PRECOMPUTED_CSVS_DIR = DATA_DIR / "precomputed_csvs"
PRECOMPUTED_CSVS_DIR.mkdir(parents=True, exist_ok=True)

TASK_FOLDERS = {
    "task1_SR": PRECOMPUTED_CSVS_DIR / "task1_SR",
    "task2_NR": PRECOMPUTED_CSVS_DIR / "task2_NR",
    "task3_TSR": PRECOMPUTED_CSVS_DIR / "task3_TSR",
}
for p in TASK_FOLDERS.values():
    p.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ZUCO_ROOT = PROJECT_ROOT / "dataset" / "ZuCo" / "q3zws" / "osfstorage"

# Input CSV Paths from Existing Pipeline
DEFAULT_PROFILE_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "cognitive_profile_output.csv"
FALLBACK_PREDICTIONS_CSV = PROJECT_ROOT / "test_nrei_csv" / "test_predictions.csv"
PERSONAL_RANGES_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "personal_reference_ranges.csv"
CLUSTER_INTERPRETATION_CSV = PROJECT_ROOT / "cognitive_profile_results_type_2" / "cluster_cognitive_interpretation.csv"
LATEST_PARTICIPANT_FILE = OUTPUT_DIR / "latest_participant.txt"


def list_zuco_tasks() -> List[Dict[str, Any]]:
    """
    Return available study reading tasks and student datasets.
    Prioritizes self-contained precomputed CSVs in neuroread_rag/data/precomputed_csvs/
    so the system runs completely independent of external .mat files and datasets.
    """
    task_order = ["task3_TSR", "task2_NR", "task1_SR"]
    task_labels = {
        "task3_TSR": "Study Material 3: Task-Specific Reading (TSR)",
        "task2_NR": "Study Material 2: Natural Reading (NR)",
        "task1_SR": "Study Material 1: Semantic Reading (SR)"
    }
    task_dir_aliases = {
        "task3 - TSR": "task3_TSR",
        "task2 - NR": "task2_NR",
        "task1- SR": "task1_SR",
        "task1 - SR": "task1_SR"
    }

    tasks = []
    seen_tasks = set()

    # 1. First scan self-contained precomputed CSV folders inside neuroread_rag/data/precomputed_csvs/
    for t_key in task_order:
        t_folder = TASK_FOLDERS.get(t_key, PRECOMPUTED_CSVS_DIR / t_key)
        csv_files = []
        if t_folder.is_dir():
            # Filter to canonical files (prefer features_*.csv, ignore duplicate symlinks)
            all_csvs = sorted(t_folder.glob("*.csv"), key=lambda p: p.name.lower())
            feature_csvs = [f for f in all_csvs if f.name.startswith("features_")]
            csv_files = feature_csvs if feature_csvs else [f for f in all_csvs if not f.is_symlink()]

        if csv_files:
            patients = []
            for cf in csv_files:
                stem = cf.stem
                # Clean patient code from filename
                # formats: features_task3_TSR_ZJS.csv, resultsZJS_TSR.csv, ZJS.csv
                code = stem
                for prefix in [f"features_{t_key}_", "features_", "results"]:
                    if code.startswith(prefix):
                        code = code[len(prefix):]
                code = code.split("_")[0]
                patients.append({
                    "filename": cf.name,
                    "patient_id": f"results{code}_{t_key.split('_')[-1]}",
                    "student_id": f"results{code}_{t_key.split('_')[-1]}",
                    "patient_code": code,
                    "student_code": code,
                    "csv_path": str(cf.resolve()),
                    "full_path": str(cf.resolve()),
                    "is_precomputed": True
                })
            tasks.append({
                "task_id": t_key,
                "task_name": task_labels.get(t_key, t_key),
                "folder": str(t_folder.resolve()),
                "patient_count": len(patients),
                "student_count": len(patients),
                "patients": patients,
                "source_type": "precomputed_csv"
            })
            seen_tasks.add(t_key)

    # 2. If raw ZuCo .mat folder exists, incorporate any tasks or uncomputed patients
    if ZUCO_ROOT.exists():
        for t_dir in sorted(ZUCO_ROOT.glob("task*")):
            norm_key = task_dir_aliases.get(t_dir.name, t_dir.name.replace(" ", "").replace("-", "_").strip("_"))
            mat_folder = t_dir / "Matlab files"
            if not mat_folder.is_dir():
                continue

            mat_files = sorted(mat_folder.glob("*.mat"), key=lambda p: p.name.lower())
            
            # If task not seen or missing precomputed patients, add/supplement
            existing_task = next((t for t in tasks if t["task_id"] == norm_key), None)
            if existing_task is None:
                patients = []
                for mf in mat_files:
                    stem = mf.stem
                    code = stem.replace("results", "").split("_")[0]
                    patients.append({
                        "filename": mf.name,
                        "patient_id": stem,
                        "student_id": stem,
                        "patient_code": code,
                        "student_code": code,
                        "full_path": str(mf.resolve()),
                        "is_precomputed": False
                    })
                tasks.append({
                    "task_id": norm_key,
                    "task_name": task_labels.get(norm_key, t_dir.name),
                    "folder": str(mat_folder.resolve()),
                    "patient_count": len(patients),
                    "student_count": len(patients),
                    "patients": patients,
                    "source_type": "matlab_raw"
                })

    # Sort tasks according to canonical order
    def task_sort(t):
        tid = t["task_id"]
        return task_order.index(tid) if tid in task_order else 99

    return sorted(tasks, key=task_sort)


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
    2. Otherwise, returns the most recently modified *.csv in PRECOMPUTED_CSVS_DIR subfolders.
    3. Otherwise, returns the most recently modified *.csv in CSV_DIR.
    4. Otherwise, returns the most recently modified *_features.csv in OUTPUT_DIR.
    5. Otherwise, falls back to DEFAULT_PROFILE_CSV or FALLBACK_PREDICTIONS_CSV.
    """
    if LATEST_PARTICIPANT_FILE.exists():
        try:
            saved_str = LATEST_PARTICIPANT_FILE.read_text(encoding="utf-8").strip()
            saved_path = Path(saved_str)
            if saved_path.exists():
                return saved_path
        except Exception:
            pass

    # Check for CSVs in PRECOMPUTED_CSVS_DIR
    if PRECOMPUTED_CSVS_DIR.exists():
        precomputed_csvs = sorted(
            PRECOMPUTED_CSVS_DIR.rglob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if precomputed_csvs:
            return precomputed_csvs[0]

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
    Find existing CSV for a participant by ID, code, or task name.
    Searches in PRECOMPUTED_CSVS_DIR (task1_SR, task2_NR, task3_TSR), CSV_DIR, or OUTPUT_DIR.
    Matches e.g. 'ZJS', 'resultsZJS_TSR', 'task3_TSR_ZJS', etc.
    """
    if not identifier:
        return None
    clean_id = identifier.replace(".mat", "").replace(".csv", "").strip().lower()

    # 1. Search in PRECOMPUTED_CSVS_DIR subfolders (task1_SR, task2_NR, task3_TSR)
    if PRECOMPUTED_CSVS_DIR.exists():
        # Exact stem match first
        for f in sorted(PRECOMPUTED_CSVS_DIR.rglob("*.csv")):
            stem_lower = f.stem.lower()
            if clean_id == stem_lower or clean_id == stem_lower.replace("features_", ""):
                return f
        # Substring match
        for f in sorted(PRECOMPUTED_CSVS_DIR.rglob("*.csv")):
            if clean_id in f.stem.lower():
                return f

    # 2. Search in dedicated CSV_DIR
    if CSV_DIR.exists():
        for f in sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            f_stem_lower = f.stem.lower()
            if clean_id in f_stem_lower:
                return f

    # 3. Search in OUTPUT_DIR
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
