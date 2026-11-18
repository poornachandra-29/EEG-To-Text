"""
ZuCo Reading Comprehension Outcome Labeling Module.
Maps external control question answers from ZuCo Task 2 (NR) to evaluate reading comprehension:
- 0: Correct response to control question
- 1: Incorrect response to control question
- NaN: Unqueried sentence (~88% of natural reading sentences)
"""

import glob
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RELATIONS_CSV = BASE_DIR / "dataset" / "ZuCo" / "q3zws" / "osfstorage" / "task2 - NR" / "Preprocessed" / "relations_normal_reading.csv"
ANSWERS_DIR = BASE_DIR / "dataset" / "ZuCo" / "q3zws" / "osfstorage" / "answers"


def extract_zuco_pid(val: Any) -> Optional[str]:
    """Extract standard ZuCo subject ID (e.g., 'ZPH') from strings like 'resultsZPH_NR'."""
    if not val or pd.isna(val):
        return None
    m = re.search(r"Z[A-Z]{2}", str(val))
    return m.group(0) if m else None


class ComprehensionLabeler:
    """
    Computes and attaches comprehension outcome labels based on ZuCo control question responses.
    """

    def __init__(self, relations_csv: Optional[Path] = None, answers_dir: Optional[Path] = None):
        self.relations_csv = Path(relations_csv or RELATIONS_CSV)
        self.answers_dir = Path(answers_dir or ANSWERS_DIR)
        self._relations_df: Optional[pd.DataFrame] = None
        self._consensus_answers: Optional[Dict[Tuple[int, int], int]] = None
        self._all_answers: Optional[pd.DataFrame] = None

    def _load_data(self) -> None:
        if self._relations_df is not None:
            return

        if not self.relations_csv.exists() or not self.answers_dir.exists():
            return

        # Load relations CSV (latin1 encoded)
        self._relations_df = pd.read_csv(
            self.relations_csv,
            sep=";",
            encoding="latin1",
            on_bad_lines="skip",
        )

        # Load participant answers across all participants
        records = []
        for mat_path in sorted(self.answers_dir.glob("Fullresults_NR_*.mat")):
            pid_raw = mat_path.stem.split("_")[-1]
            pid = extract_zuco_pid(pid_raw) or pid_raw
            try:
                mat = sio.loadmat(mat_path)
                if "answers" in mat:
                    for row in mat["answers"]:
                        records.append({
                            "pid": pid,
                            "para": int(row[0]),
                            "sent": int(row[1]),
                            "choice": int(row[2]),
                        })
            except Exception:
                continue

        if records:
            self._all_answers = pd.DataFrame(records)
            self._consensus_answers = {}
            for (para, sent), grp in self._all_answers.groupby(["para", "sent"]):
                mode_choice = grp["choice"].mode().iloc[0]
                self._consensus_answers[(para, sent)] = int(mode_choice)

    def attach_labels(self, df: pd.DataFrame, pid: Optional[str] = None) -> pd.DataFrame:
        """
        Add 'comprehension_error' column to DataFrame.
        Handles both single-subject DataFrames and multi-subject concatenated DataFrames.
        """
        out = df.copy()
        out["comprehension_error"] = np.nan

        self._load_data()
        if self._relations_df is None or self._consensus_answers is None or self._all_answers is None:
            return out

        # Determine subject column or list of subjects
        if pid:
            subjs = [extract_zuco_pid(pid) or pid]
            subj_series = pd.Series(subjs[0], index=out.index)
        elif "patient_id" in out.columns:
            subj_series = out["patient_id"].apply(extract_zuco_pid)
        elif "patient_name" in out.columns:
            subj_series = out["patient_name"].apply(extract_zuco_pid)
        else:
            subj_series = pd.Series(None, index=out.index)

        # Cache text lookup to (para, sent)
        text_to_para_sent = {}
        for _, rel_row in self._relations_df.iterrows():
            sent_str = str(rel_row.get("sentence", "")).strip()[:35]
            if sent_str:
                text_to_para_sent[sent_str] = (int(rel_row["paragraph_id"]), int(rel_row["sentence_id"]))

        error_col = np.full(len(out), np.nan)

        # Group by participant
        unique_subjs = [s for s in subj_series.unique() if s is not None]
        for s in unique_subjs:
            s_mask = (subj_series == s).values
            p_ans_df = self._all_answers[self._all_answers["pid"] == s]
            if p_ans_df.empty:
                continue
            p_ans = p_ans_df.set_index(["para", "sent"])["choice"].to_dict()

            sub_indices = np.where(s_mask)[0]
            for idx in sub_indices:
                txt = str(out["sentence_text"].iloc[idx]).strip()[:35]
                if txt in text_to_para_sent:
                    key = text_to_para_sent[txt]
                    if key in p_ans and key in self._consensus_answers:
                        u_c = p_ans[key]
                        t_c = self._consensus_answers[key]
                        error_col[idx] = 1.0 if u_c != t_c else 0.0

        out["comprehension_error"] = error_col
        return out


def attach_comprehension_labels(df: pd.DataFrame, pid: Optional[str] = None) -> pd.DataFrame:
    labeler = ComprehensionLabeler()
    return labeler.attach_labels(df, pid=pid)
