# 📋 Model Card: NeuroRead RAG (v2.0)

## Model Details
- **Developer**: NeuroRead Development Team / SPLAB
- **Model Date**: November 2026
- **Model Version**: 2.0.0
- **Model Type**: Multi-modal physiological & lexical analytics pipeline integrating 128-channel EEG (GSN-HydroCel-128) and eye-tracking (EyeLink 1000) for reading assessment and cognitive pacing.
- **License**: MIT / Academic Research Use

---

## Intended Use
- **Primary Intended Uses**:
  - Educational self-reflection for adult learners to identify comprehension bottlenecks.
  - Pacing optimization (session fatigue changepoints and empirical Pomodoro calibration).
  - Research evaluation of reading difficulty independent of lexical and syntactic complexity.
- **Out-of-Scope / Prohibited Uses**:
  - Clinical diagnosis of learning disabilities (e.g., dyslexia, ADHD).
  - High-stakes academic testing or student evaluation.
  - Deployment on unseen age cohorts (children, adolescents) without dedicated re-calibration.

---

## Training & Validation Data
- **Corpus**: ZuCo 1.0 (Zurich Cognitive Language Processing Corpus, Hollenstein et al., 2018).
- **Participants**: 12 healthy adult native English speakers (5 female, 7 male, ages 22–54, all right-handed).
- **Trials**: 3,493 natural reading trials across 300 Wikipedia biographical sentences (Task 2 NR).
- **Validation Protocol**: Leave-One-Subject-Out (LOSO) cross-validation across all 12 participants and GroupKFold grouped by sentence text.

---

## Performance & Scientific Findings
1. **Confound Control**: Raw physiological metrics correlate strongly with text length and surprisal (Pearson $r \approx 0.37$). After Leave-One-Subject-Out Ridge residualization, residual correlations with all lexical covariates are $|r| < 0.001$, isolating reader-specific cognitive effort.
2. **Behavioral Breakdown Prediction (`breakdown_behav`)**:
   - Text Covariates Alone: AUROC **0.495** [0.469, 0.516] (random chance, confirming confounds are removed).
   - EEG Power Bands Alone: AUROC **0.561** [0.536, 0.583].
   - Non-Leaking Eye-Tracking (FFD, PupilSize): AUROC **0.636** [0.614, 0.658].
   - Combined EEG + Eye-Tracking: AUROC **0.642** [0.620, 0.665].
3. **External Comprehension Errors (`comprehension_error`)**:
   - 416 control question trials evaluated across 12 participants (12.0% error rate).
   - Instantaneous sentence-level physiology yields AUROC **0.556**, indicating that natural reading comprehension errors largely reflect macro-level conceptual integration or background knowledge rather than isolated sentence ocular stalls.
4. **Clustering & Stability**:
   - Dimensionality reduction via PCA retaining 93.4% variance.
   - Sweep over $K \in [2, 10]$ across full, diagonal, and tied covariances.
   - Model selection enforces a stability floor of $\text{ARI} \ge 0.60$ over 20 random seeds.
   - $K=3$ (Full Covariance) achieves stability ARI **0.991** with BIC **64,912.9**. Higher $K$ solutions ($K=4..8$) fail the stability floor (ARI $0.38 - 0.57$).
   - Sentences with posterior confidence $< 0.50$ (14.8%) are labeled as `Indeterminate`.

---

## Ethical & Transparency Considerations
- **Non-Diagnostic Nature**: All composite indices (LES, CWI, REI, SAI) are explicitly tagged as empirical heuristic proxies.
- **Uncertainty & Small Sample Size**: Evaluations report bootstrap 95% confidence intervals reflecting $n=12$ participant limitations.
- **Zero Fabrication**: Non-significant signals and modest effect sizes are documented without selective cherry-picking.
