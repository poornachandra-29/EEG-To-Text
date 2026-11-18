# 🧠 NeuroRead RAG (v2.0): Research-Grounded Cognitive Reading Analytics

A scientific, multi-modal Retrieval-Augmented Generation (RAG) platform and deterministic physiological analytics engine designed for the Zurich Cognitive Language Processing Corpus (ZuCo 1.0; simultaneous 128-channel EEG and eye-tracking).

---

## 🌟 Key Capabilities & Architectural Upgrades (v2.0)

* **Lexical & Syntactic Confound Control (Phase 2)**: Removes predictable text difficulty (sentence length, Zipf word frequency, surprisal, dependency syntax) using Leave-One-Subject-Out (LOSO) cross-fitted Ridge regression. Flags indicate reading friction *relative to what text complexity predicts*, not general sentence hardness.
* **Empirical Behavioral Breakdown (Phase 3)**: Eliminates fixed 25% quartile quotas. Behavioral breakdowns (`breakdown_behav`) are defined by standardized residualized processing effort (`TRT_rz >= 1.0` and ocular refixation). Legacy quartile categories are retained for descriptive comparison under `legacy_state`.
* **External Comprehension Validation (Phase 3 & 4)**: Maps ZuCo Task 2 control questions to evaluate objective comprehension error rates across 416 trials with empirical validation and bootstrap confidence intervals.
* **Frontal Midline Theta (FMT) Anatomical Mapping (Phase 1)**: Computes true frontal midline spectral power from GSN-HydroCel-128 electrodes closest to Fz and FCz (E11, E6; indices `[5, 10]`), isolating true executive control signals.
* **Spectral Reliability Gating (Phase 1)**: Duration-aware Welch PSD estimation:
  - $\ge 4.0$ s: Full reliability (`nperseg = 512`)
  - $2.0 \le t < 4.0$ s: Low reliability (`nperseg = 256`)
  - $< 2.0$ s: Insufficient signal (`NaN` spectral features, preventing zero-imputation artifacts).
* **Multi-Modal Predictive Validation & Pruning (Phase 4)**: Cross-validated under Leave-One-Subject-Out (LOSO) across all 12 participants. Prunes collinear redundant metrics ($|r| > 0.85$, VIF $> 10$) and generates `reports/metric_importance.csv` and `reports/validation_report.md`.
* **Clustering Rework & Stability Floor (Phase 5)**: Dimensionality reduction via PCA (93.4% variance explained) and full sweep over $K \in [2, 10]$ across full, diagonal, and tied covariances. Enforces a stability floor of mean pairwise $\text{ARI} \ge 0.60$ over 20 random seeds. Identifies $K=3$ (Full Covariance) as the optimal stable solution (Stability $\text{ARI} = 0.991$, BIC = 64,912.9). Ambiguous trials with max posterior $< 0.50$ (14.8%) are labeled `Indeterminate`.
* **Real Timing & Fatigue Changepoints (Phase 8)**: Replaces synthetic multipliers with true elapsed recording time derived from sensor durations. Fits two-segment piecewise linear regression to identify session cognitive fatigue onsets, providing empirically calibrated Pomodoro study blocks.
* **Transparent Heuristic Labeling (Phase 6)**: All composite scores (Learning Efficiency Score, Cognitive Workload Index, Reading Engagement Index) are explicitly tagged as empirical heuristic proxies for learner reflection.

---

## 📂 Repository Structure

```
EEG-To-Text/
├── config.yaml                       # Global configuration (sampling rates, band edges, thresholds, seeds)
├── CHANGELOG.md                      # Detailed release history and phase breakdown
├── CITATION.cff                      # Academic citation metadata
├── docs/
│   ├── DATA_NOTES.md                 # ZuCo 1.0 data inspection, montage mapping, control questions
│   └── MODEL_CARD.md                 # Standardized Model Card (Mitchell et al., 2019)
├── figures/
│   ├── eeg_reliability_by_sentence_length.png
│   ├── trt_vs_surprisal_residualization.png
│   └── cluster_stability_sweep.png
├── reports/
│   ├── validation_report.md          # Multi-group LOSO evaluation and statistical report
│   ├── metric_importance.csv         # Standardized coefficients, odds ratios, and VIFs
│   └── cluster_selection.csv         # Full BIC, AIC, and stability sweep over K=2..8
└── neuroread_rag/
    ├── schemas.py                    # Typed Pydantic stage schemas
    ├── config.py                     # Metric definitions, constants, and path helpers
    ├── main.py                       # CLI entrypoint for report generation and server launch
    ├── server.py                     # API server for interactive dashboard
    ├── participant_analysis/
    │   ├── feature_extractor.py      # Welch PSD, reliability gating, artifact rejection
    │   ├── text_features.py          # Surprisal, Zipf frequency, syntax depth
    │   ├── residualize.py            # LOSO Ridge confound residualizer & robust z-scores
    │   ├── comprehension_labels.py   # Control question mapping & consensus derivation
    │   ├── sentence_comparator.py    # Breakdown classifier & fatigue curve
    │   └── analyzer.py               # Session analytics, baselines, composite indices
    ├── cluster_machine_learning/
    │   └── cluster_rework.py         # PCA + GMM sweep, stability floor, cluster profiles
    ├── validation/
    │   └── predictive_models.py      # Cross-modal LOSO evaluation, bootstrap CIs, changepoints
    └── tests/
        ├── snapshots/legacy_ZPH.json # Legacy baseline output for non-regression parity
        ├── test_participant_analysis.py
        └── test_student_learning.py
```

---

## 🚀 Quickstart & Reproducibility

### 1. Run All Tests
```bash
.venv/bin/python -m unittest discover neuroread_rag/tests/
```

### 2. Run End-to-End Pipeline
```bash
.venv/bin/python -m neuroread_rag.run_all
```

### 3. Launch Interactive Dashboard
```bash
.venv/bin/python -m neuroread_rag.server --port 8000
```
Open `http://localhost:8000` in your web browser.
