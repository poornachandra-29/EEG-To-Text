# 📝 Changelog: NeuroRead RAG Pipeline

All notable changes to the NeuroRead RAG package are documented here.

## [2.0.0] - 2026-11-15

### Phase 0: Baseline, Config, Schemas & Testing
- Added `config.yaml` specifying sampling rates, channel selections, thresholds, model hyperparameters, and seed controls. Loaded globally.
- Implemented typed Pydantic schemas in `schemas.py` (`SentenceFeatures`, `SentenceLabels`, `Finding`, `Recommendation`, `Report`).
- Added multi-library deterministic seed utility `neuroread_rag/utils/seeds.py` (`set_seed(42)`).
- Generated baseline legacy report snapshot `tests/snapshots/legacy_ZPH.json` (151 KB).
- Added `docs/DATA_NOTES.md` documenting ZuCo 1.0 electrode layout (GSN-HydroCel-128, 105 retained channels), duration gating, and control questions.

### Phase 1: Feature Extraction Overhaul
- **Frontal Midline Theta (FMT)**: Replaced whole-scalp 105-channel average with anatomical frontal midline electrode pair: E6 (FCz) and E11 (Fz), mapped to indices `[5, 10]`. Added `theta_fm` and `theta_global`.
- **Duration Reliability Gating**: Added minimum duration checks for spectral estimation:
  - $\ge 4.0$ s: `eeg_reliability = "full"` (`nperseg = 512`)
  - $2.0 \le t < 4.0$ s: `eeg_reliability = "low"` (`nperseg = 256`)
  - $< 2.0$ s: `eeg_reliability = "insufficient"`, spectral features set to `NaN` without zero imputation.
- **Artifact Fraction**: Epochs with peak-to-peak amplitude $> 150 \mu\text{V}$ flagged; sentences exceeding 20% rejected samples marked `insufficient`.
- **Log Band Powers**: Log10 power transforms added to resolve skew in raw spectral powers.
- Generated diagnostic figure `figures/eeg_reliability_by_sentence_length.png`.

### Phase 2: Lexical & Syntactic Confound Control
- Implemented `text_features.py` extracting sentence length, Zipf frequency, surprisal, dependency depth, clauses, entity counts, and readability scores.
- Implemented `residualize.py` using Leave-One-Subject-Out (LOSO) cross-fitted Ridge regression.
- Residualized physiological features (`<feat>_resid`) and computed within-subject robust z-scores (`<feat>_rz`).
- Verified residual orthogonality ($|r| < 0.001$ against text covariates).
- Generated diagnostic figure `figures/trt_vs_surprisal_residualization.png`.

### Phase 3: New Label Definitions & Leakage Guardrails
- **Behavioral Breakdown (`breakdown_behav`)**: Replaced fixed 25% quartile quota with confound-controlled threshold: `TRT_rz >= 1.0` and (`FixationCount_rz >= 0.5` or `GD_rz >= 0.5`).
- **Continuous Excess Processing**: Added `excess_processing = mean(TRT_rz, FixationCount_rz, GD_rz)`.
- **External Comprehension Errors (`comprehension_error`)**: Mapped ZuCo Task 2 control questions to participant responses across 416 trials.
- **Legacy Label Preservation**: Preserved legacy quartile states in `legacy_state` (labeled descriptive only) and `reading_state`.
- **Leakage Guards**: Implemented `verify_no_leakage()` strictly blocking defining features from predictor sets during modeling.

### Phase 4: Cross-Modal Predictive Validation & Metric Pruning
- Evaluated models across 5 modality groups using Leave-One-Subject-Out (LOSO) and GroupKFold cross-validation.
- Calculated bootstrap 95% confidence intervals and permutation significance p-values.
- Computed Variance Inflation Factors (VIF) and pairwise correlations to prune redundant collinear variants (`FMT`, `ThetaPower`, `ASM`).
- Generated `reports/metric_importance.csv` and `reports/validation_report.md`.
- Implemented two-segment piecewise linear regression selected by BIC to detect session cognitive fatigue onset.

### Phase 5: Clustering Architecture Rework
- Standardized pruned physiological features and reduced dimensionality via PCA (retaining 93.4% variance).
- Swept $K \in [2, 10]$ across full, diagonal, and tied covariances over 20 random seeds.
- Enforced stability floor $\text{ARI} \ge 0.60$. Empirically selected $K=3$ (Full Covariance) with stability ARI **0.991** and BIC **64,912.9**.
- Flagged sentences with posterior probability $< 0.50$ (14.8%) as `Indeterminate`.
- Generated `reports/cluster_selection.csv`, `artifacts/cluster_profiles.json`, and `figures/cluster_stability_sweep.png`.

### Phase 6: Composite Indices & Heuristic Labeling
- Calibrated weights for Reading Engagement Index (REI) and Cognitive Workload Index (CWI) using standardized regression coefficients.
- Implemented rolling 10-sentence Sustained Attention Index (SAI) formula with edge boundary padding.
- Explicitly tagged all composite indices and LES tiers as heuristic proxies for self-directed study reflection.

### Phase 8: Chronological Fatigue Curve & Real Timing
- Replaced synthetic `sentence_index * 0.12` multiplier with true elapsed session time derived from sensor recording durations (`eeg_duration_s`, `TRT`).
- Implemented Words Per Minute (WPM) tracking across session checkpoints.
- Clamped Pomodoro recommendation to $[15, 40]$ minutes rounded to nearest 5 minutes, with fallback for sessions without detected fatigue.

### Phase 9: Calibration & Statistical Hygiene
- Implemented Winsorization at 1st and 99th percentiles for heavy-tailed eye-tracking and pupil distributions.
- Validated robust z-score formulas with fallbacks to IQR / 1.349 and sample standard deviation.
- Documented transductive session baseline standards.
