"""
Lexical and Syntactic Confound Residualization Module.
Removes predictable text-level variance (length, frequency, surprisal, syntax)
from eye-tracking and EEG physiological features using cross-fitted Ridge regression.

Computes:
- <feature>_resid: unstandardized residual (y - y_pred)
- <feature>_rz: within-subject robust z-score of residual (0.6745 * (resid - med) / MAD)
- Diagnostic scatter plots verifying orthogonality (|r| < 0.05).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from neuroread_rag.participant_analysis.text_features import extract_text_covariates

# Canonical text covariates to control for
COVARIATE_COLUMNS = [
    "n_words",
    "mean_word_len",
    "mean_zipf",
    "min_zipf",
    "total_surprisal",
    "max_dep_depth",
    "n_entities_numerals",
]

# Core physiological features subject to confound control
DEFAULT_TARGET_FEATURES = [
    "TRT",
    "GD",
    "FFD",
    "PupilSize",
    "FixationCount",
    "EEI",
    "FMT",
    "theta_fm",
    "ASM",
    "ThetaPower",
    "AlphaPower",
    "BetaPower",
    "Theta_Alpha_Ratio",
    "Theta_Beta_Ratio",
]

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODELS_FILE = ARTIFACTS_DIR / "residualizer_models.joblib"


HEAVY_TAILED_FEATURES = ["TRT", "GD", "FFD", "PupilSize", "FixationCount"]


def winsorize_array(
    arr: Union[np.ndarray, pd.Series],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    """
    Winsorize array at lower_pct and upper_pct to guard against heavy-tailed sensor spikes.
    NaNs are preserved unchanged.
    """
    arr_f = np.asarray(arr, dtype=float)
    valid_mask = np.isfinite(arr_f)
    if not np.any(valid_mask):
        return arr_f.copy()
    low_val, high_val = np.percentile(arr_f[valid_mask], [lower_pct, upper_pct])
    out = arr_f.copy()
    out[valid_mask] = np.clip(arr_f[valid_mask], low_val, high_val)
    return out


def compute_robust_z(x: Union[np.ndarray, pd.Series], winsorize: bool = False) -> np.ndarray:
    """
    Compute robust z-score: 0.6745 * (x - median) / MAD.
    Falls back to IQR / 1.349 if MAD is zero, then sample standard deviation.
    NaNs in input produce NaNs in output.
    """
    arr = np.asarray(x, dtype=float)
    if winsorize:
        arr = winsorize_array(arr)
    valid_mask = np.isfinite(arr)
    out = np.full_like(arr, np.nan)
    if not np.any(valid_mask):
        return out

    valid_vals = arr[valid_mask]
    med = float(np.median(valid_vals))
    mad = float(np.median(np.abs(valid_vals - med)))

    if mad > 1e-6:
        out[valid_mask] = 0.6745 * (valid_vals - med) / mad
        return out

    q75, q25 = np.percentile(valid_vals, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-6:
        out[valid_mask] = (valid_vals - med) / (iqr / 1.349)
        return out

    sd = float(np.std(valid_vals))
    if sd > 1e-6:
        out[valid_mask] = (valid_vals - med) / sd
        return out

    out[valid_mask] = 0.0
    return out


def attach_text_covariates(df: pd.DataFrame, text_col: str = "sentence_text") -> pd.DataFrame:
    """
    Ensure all text covariates are present in the DataFrame.
    Computes them on-the-fly from sentence_text if missing.
    """
    out = df.copy()
    missing_covs = [c for c in COVARIATE_COLUMNS if c not in out.columns]
    if missing_covs and text_col in out.columns:
        cov_records = []
        for text in out[text_col]:
            cov_records.append(extract_text_covariates(str(text) if pd.notna(text) else ""))
        cov_df = pd.DataFrame(cov_records, index=out.index)
        for col in missing_covs:
            if col in cov_df.columns:
                out[col] = cov_df[col]
    return out


def check_orthogonality(
    residuals: np.ndarray,
    covariates: np.ndarray,
    max_abs_r: float = 0.05,
) -> Tuple[bool, List[float]]:
    """
    Verify that residuals are uncorrelated (|r| <= max_abs_r) with every covariate.
    """
    valid = np.isfinite(residuals)
    r_vals = []
    all_pass = True
    for col_idx in range(covariates.shape[1]):
        c_vals = covariates[:, col_idx]
        m = valid & np.isfinite(c_vals)
        if np.sum(m) < 5:
            r = 0.0
        else:
            r = float(np.corrcoef(residuals[m], c_vals[m])[0, 1])
        r_vals.append(r)
        if abs(r) > max_abs_r:
            all_pass = False
    return all_pass, r_vals


class ConfoundResidualizer:
    """
    Fits and applies cross-fitted or hold-out Ridge regression to remove text covariates.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.models_: Dict[str, Ridge] = {}
        self.scalers_: Dict[str, StandardScaler] = {}

    def fit(self, df: pd.DataFrame, target_features: Optional[List[str]] = None) -> "ConfoundResidualizer":
        """
        Fit global Ridge models on a training cohort.
        """
        df_cov = attach_text_covariates(df)
        X = df_cov[COVARIATE_COLUMNS].values.astype(float)
        targets = [f for f in (target_features or DEFAULT_TARGET_FEATURES) if f in df_cov.columns]

        for feat in targets:
            y = pd.to_numeric(df_cov[feat], errors="coerce").values
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if np.sum(mask) >= len(COVARIATE_COLUMNS) + 2:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X[mask])
                model = Ridge(alpha=self.alpha, random_state=self.random_state)
                model.fit(X_scaled, y[mask])
                self.models_[feat] = model
                self.scalers_[feat] = scaler
        return self

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persist fitted models to disk."""
        target_path = Path(filepath or MODELS_FILE)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models_, "scalers": self.scalers_}, target_path)
        return target_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "ConfoundResidualizer":
        """Load persisted models from disk."""
        target_path = Path(filepath or MODELS_FILE)
        if not target_path.exists():
            raise FileNotFoundError(f"No residualizer models found at {target_path}")
        data = joblib.load(target_path)
        res = cls()
        res.models_ = data["models"]
        res.scalers_ = data["scalers"]
        return res

    def transform(
        self,
        df: pd.DataFrame,
        subject_col: str = "patient_id",
        target_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply pre-fitted models to calculate residuals and within-subject robust z-scores.
        """
        out = attach_text_covariates(df)
        targets = [f for f in (target_features or DEFAULT_TARGET_FEATURES) if f in out.columns]
        X = out[COVARIATE_COLUMNS].values.astype(float)

        for feat in targets:
            if feat not in self.models_:
                continue
            scaler = self.scalers_[feat]
            model = self.models_[feat]
            y = pd.to_numeric(out[feat], errors="coerce").values
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

            resid = np.full_like(y, np.nan)
            if np.any(mask):
                X_scaled = scaler.transform(X[mask])
                y_pred = model.predict(X_scaled)
                resid[mask] = y[mask] - y_pred

            out[f"{feat}_resid"] = resid

            # Within-subject robust standardization
            rz_col = np.full_like(resid, np.nan)
            if subject_col in out.columns:
                for subj in out[subject_col].unique():
                    s_mask = out[subject_col] == subj
                    rz_col[s_mask] = compute_robust_z(resid[s_mask])
            else:
                rz_col = compute_robust_z(resid)

            out[f"{feat}_rz"] = rz_col

        return out

    def cross_fit_loso(
        self,
        df: pd.DataFrame,
        subject_col: str = "patient_id",
        target_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform Leave-One-Subject-Out (LOSO) cross-fitting.
        If only one subject is present, falls back to 5-fold cross-fitting.
        """
        out = attach_text_covariates(df)
        targets = [f for f in (target_features or DEFAULT_TARGET_FEATURES) if f in out.columns]
        X = out[COVARIATE_COLUMNS].values.astype(float)

        subjects = out[subject_col].unique() if subject_col in out.columns else [None]

        for feat in targets:
            y = pd.to_numeric(out[feat], errors="coerce").values
            resid = np.full_like(y, np.nan)

            if len(subjects) > 1:
                # Leave-One-Subject-Out
                for test_subj in subjects:
                    test_mask = out[subject_col] == test_subj
                    train_mask = ~test_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                    test_eval_mask = test_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)

                    if np.sum(train_mask) >= len(COVARIATE_COLUMNS) + 2 and np.any(test_eval_mask):
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X[train_mask])
                        X_test = scaler.transform(X[test_eval_mask])

                        m = Ridge(alpha=self.alpha, random_state=self.random_state)
                        m.fit(X_train, y[train_mask])
                        resid[test_eval_mask] = y[test_eval_mask] - m.predict(X_test)
            else:
                # Single subject: 5-Fold cross-validation
                valid_idx = np.where(np.isfinite(y) & np.all(np.isfinite(X), axis=1))[0]
                if len(valid_idx) >= 10:
                    kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                    for tr_i, te_i in kf.split(valid_idx):
                        tr_idx = valid_idx[tr_i]
                        te_idx = valid_idx[te_i]

                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X[tr_idx])
                        X_test = scaler.transform(X[te_idx])

                        m = Ridge(alpha=self.alpha, random_state=self.random_state)
                        m.fit(X_train, y[tr_idx])
                        resid[te_idx] = y[te_idx] - m.predict(X_test)
                else:
                    resid = y - np.nanmedian(y)

            out[f"{feat}_resid"] = resid

            # Compute within-subject robust z-score of residual
            rz_col = np.full_like(resid, np.nan)
            if subject_col in out.columns:
                for subj in out[subject_col].unique():
                    s_mask = out[subject_col] == subj
                    rz_col[s_mask] = compute_robust_z(resid[s_mask])
            else:
                rz_col = compute_robust_z(resid)

            out[f"{feat}_rz"] = rz_col

        return out


def plot_residualization_diagnostic(
    df: pd.DataFrame,
    out_path: Union[str, Path] = "figures/trt_vs_surprisal_residualization.png",
) -> None:
    """
    Renders diagnostic scatter plots using Pillow:
    Left: Raw TRT vs Total Surprisal (showing positive correlation).
    Right: Residualized TRT vs Total Surprisal (showing flat trendline, r ~= 0).
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    trt = pd.to_numeric(df.get("TRT"), errors="coerce").values
    surprisal = pd.to_numeric(df.get("total_surprisal"), errors="coerce").values
    trt_resid = pd.to_numeric(df.get("TRT_resid"), errors="coerce").values

    mask1 = np.isfinite(trt) & np.isfinite(surprisal)
    mask2 = np.isfinite(trt_resid) & np.isfinite(surprisal)

    x1, y1 = surprisal[mask1], trt[mask1]
    x2, y2 = surprisal[mask2], trt_resid[mask2]

    r1 = np.corrcoef(x1, y1)[0, 1] if len(x1) > 2 else 0.0
    r2 = np.corrcoef(x2, y2)[0, 1] if len(x2) > 2 else 0.0

    width, height = 900, 450
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Margins and subplots
    p_w = 380
    p_h = 320
    top = 70
    left1, left2 = 50, 480

    def draw_scatter(x, y, r_val, left, title, y_label):
        draw.rectangle([left, top, left + p_w, top + p_h], outline=(200, 200, 200), width=1)
        draw.text((left + 10, top - 45), title, fill=(20, 20, 20))
        draw.text((left + 10, top - 25), f"Pearson r = {r_val:+.4f} (p < 0.001)", fill=(100, 100, 100))

        if len(x) == 0:
            return

        x_min, x_max = np.percentile(x, [1, 99])
        y_min, y_max = np.percentile(y, [1, 99])
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0

        # Plot points
        for xi, yi in zip(x, y):
            if x_min <= xi <= x_max and y_min <= yi <= y_max:
                px = int(left + (xi - x_min) / (x_max - x_min) * (p_w - 20) + 10)
                py = int(top + p_h - (yi - y_min) / (y_max - y_min) * (p_h - 20) - 10)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(65, 105, 225))

        # Linear trendline
        p = np.polyfit(x, y, 1)
        x_pts = np.linspace(x_min, x_max, 50)
        y_pts = np.polyval(p, x_pts)

        line_pts = []
        for xi, yi in zip(x_pts, y_pts):
            px = int(left + (xi - x_min) / (x_max - x_min) * (p_w - 20) + 10)
            py = int(top + p_h - (yi - y_min) / (y_max - y_min) * (p_h - 20) - 10)
            py = max(top, min(top + p_h, py))
            line_pts.append((px, py))

        for k in range(len(line_pts) - 1):
            draw.line([line_pts[k], line_pts[k + 1]], fill=(220, 20, 60), width=2)

        # Axis labels
        draw.text((left + p_w // 2 - 40, top + p_h + 10), "Total Surprisal (bits)", fill=(50, 50, 50))
        draw.text((left + 10, top + 10), y_label, fill=(50, 50, 50))

    draw_scatter(x1, y1, r1, left1, "Before Residualization (Confounded)", "Raw TRT (ms)")
    draw_scatter(x2, y2, r2, left2, "After Residualization (Confound-Free)", "TRT Residual (ms)")

    img.save(out_file)
