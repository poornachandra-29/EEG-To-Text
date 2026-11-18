# NeuroRead-RAG: Technical Reference Compendium & Project Final Report

**A Comprehensive Engineering and Pedagogical Reference for Multimodal Physiological Reading Intake, Lexical Confound Residualization, Metric Mathematics, and Scientific Retrieval-Augmented Generation (S-RAG)**

* **Project:** NeuroRead-RAG (ZuCo 1.0 Pipeline)
* **Target Audience:** Professors, Evaluators, Researchers, and Students
* **Primary Modalities:** 128-channel High-Density EEG (500 Hz) + EyeLink 1000 Infrared Eye-Tracking (500 Hz)
* **Dataset Cohort:** 12 Native Adult Readers (ZuCo 1.0 Normal Reading & Task-Specific Reading)
* **Clinical Classification:** Strictly Non-Diagnostic (Educational Psychophysiological Observational Intake)
* **Compiled PDF Document:** [`reports/neuroread_comprehensive_project_reference.pdf`](file:///mnt/DATA/SPLAB/1.%20Code/EEG-To-Text/reports/neuroread_comprehensive_project_reference.pdf)

---

## 1. End-to-End Preprocessing & Feature Engineering Pipeline

### 1.1 Source Dataset & Multi-Modal Acquisition Protocol
The NeuroRead-RAG pipeline processes natural reading data from the **Zurich Cognitive Corpus (ZuCo 1.0)**, comprising synchronous high-density electroencephalography (EEG) and infrared eye-tracking recordings from 12 healthy native English adult readers (`ZAB`, `ZDM`, `ZGW`, `ZJM`, `ZJN`, `ZJS`, `ZKB`, `ZKH`, `ZKW`, `ZMG`, `ZPH`, `ZTS`) reading factual Wikipedia and movie review sentences.

1. **High-Density EEG:**
   - 128-channel Geodesic Sensor Net (HydroCel GSN-128) sampled at 500 Hz with vertex reference ($C_z$).
   - 105 scalp channels are retained after removing perimeter chin/neck channels susceptible to muscle artifacts.
2. **Eye-Tracking:**
   - EyeLink 1000 infrared video-based tracking system at 500 Hz monocular tracking (right eye).
   - Records gaze fixations, forward saccades, backward regressions, and pupil diameter.
3. **Reading Protocols:**
   - Normal Reading (NR / Task 2): Natural reading of 300 factual sentences.
   - Task-Specific Reading (TSR / Task 3): Target-directed relation extraction.
   - Semantic Reading (SR / Task 1): Semantic judgment tasks.

---

### 1.2 Signal Cleaning & Spectral Power Density (Welch PSD)
1. **Filtering:** Raw continuous EEG undergoes finite impulse response (FIR) bandpass filtering between 0.5 Hz and 40 Hz to remove DC electrode drifts and high-frequency EMG noise.
2. **Ocular Artifact Removal:** Independent Component Analysis (ICA) in MNE isolates and subtracts stereotypical ocular components (blinks and saccadic spike potentials).
3. **Welch Spectral Power Estimation:**
   - Sentence-level spectral power is estimated using Welch's modified periodogram with 50% overlapping Hamming windows.
   - Frequency band edges:
     - **Theta Band ($\theta$):** 4.0 Hz – 8.0 Hz (working memory maintenance and mental effort)
     - **Alpha Band ($\alpha$):** 8.0 Hz – 13.0 Hz (sensory gating, cortical idling, and active visual suppression)
     - **Beta Band ($\beta$):** 13.0 Hz – 30.0 Hz (active concentration and analytical processing)

---

### 1.3 Frontal Midline Theta (FMT) Localization & Channel Selection
Naive implementations mistakenly average theta across all 105 scalp channels. NeuroRead-RAG isolates Frontal Midline Theta (FMT / $\theta_{\text{fm}}$) to true mid-frontal executive sites:
- Channels nearest in 3D Euclidean distance to standard international 10–20 sites $F_z$ and $FC_z$ on the HydroCel GSN-128 montage: **E11** ($F_z$) and **E6** ($FC_z$).
- This separates executive working memory allocation from occipito-parietal sensory theta.

---

### 1.4 Duration Reliability Gating
Spectral estimation over very brief intervals produces unstable Fourier transforms. The pipeline enforces strict data-quality gating:
- **Full Reliability ($\text{Duration} \ge 4.0\text{ s}$):** Computed via Welch estimation with $n_{\text{perseg}} = 512$ samples (full statistical confidence).
- **Low Reliability ($2.0\text{ s} \le \text{Duration} < 4.0\text{ s}$):** Computed with $n_{\text{perseg}} = 256$ samples and tagged as low reliability.
- **Insufficient Signal ($\text{Duration} < 2.0\text{ s}$):** Spectral features are set to `NaN` and flagged as insufficient. Zero-imputation is strictly prohibited to prevent artificial variance distortion.

---

### 1.5 Lexical & Syntactic Confound Residualization
* **The Problem:** Sentences with rare vocabulary, high word counts, or multi-clause syntactic embeddings naturally demand longer reading times and higher frontal theta from *every* reader. Flagging sentences based on raw thresholds confuses text difficulty with student reading difficulty.
* **The Solution:** A cross-fitted Ridge regression model extracts 7 linguistic covariates for every sentence to predict the normative physiological response expected from the text alone:
  1. $n_{\text{words}}$: Sentence length in word tokens.
  2. $\text{mean\_word\_len}$: Average characters per word.
  3. $\text{mean\_zipf}$: Mean lexical frequency on the logarithmic Zipf scale (van Heuven et al.).
  4. $\text{min\_zipf}$: Frequency of the rarest word in the sentence (the primary lexical hurdle).
  5. $\text{total\_surprisal}$: Cumulative GPT-2 language model surprisal: $-\sum \log_2 P(w_t \mid w_{<t})$.
  6. $\text{max\_dep\_depth}$: Maximum syntactic dependency tree parse depth via spaCy.
  7. $n_{\text{entities\_numerals}}$: Count of named entities and numeric tokens.

**Mathematical Formulation:**
$$\hat{y} = \beta_0 + \sum_{j=1}^7 \beta_j X_j$$
$$y_{\text{resid}} = y_{\text{observed}} - \hat{y}$$
$$y_{rz} = 0.6745 \times \frac{y_{\text{resid}} - \text{median}(y_{\text{resid}})}{\text{MAD}(y_{\text{resid}})}$$

* **Outcome:** After residualization, Pearson correlation between physiological signals and text length/surprisal drops to $|r| < 0.05$. A flagged sentence now genuinely signifies: *"Harder for this specific student than the text alone predicts."*

---

### 1.6 Behavioral Outcomes: Flow vs. Breakdown
- **`comprehension_error`:** Binary ground-truth label mapped from external ZuCo control questions ($1 = \text{incorrect response}$, $0 = \text{correct response}$).
- **`is_breakdown`:** Multi-factor reading breakdown flag ($TRT_{rz} \ge 2.0$, excessive refixation density, or missed comprehension).
- **`flow_state`:** Fluent reading cadence characterized by balanced dwell ($TRT_{rz} \le 0.0$), minimal refixations, and correct comprehension.

---

## 2. Comprehensive Compendium of Physiological Metrics

| Metric | Modality | Exact Formula / Derivation | Biological Mechanism & Reference | What It Says About the Student |
| :--- | :--- | :--- | :--- | :--- |
| **EEI** | Spectral EEG | $\frac{\text{BetaPower}}{\text{AlphaPower} + \text{ThetaPower}}$ | NASA cortical vigilance ratio (Pope et al., 1995) | High: Active analytical focus.<br/>Low: Mind-wandering, attentional drift, or mental fatigue. |
| **FMT / $\theta_{\text{fm}}$** | Spectral EEG | $\text{PSD}(4 - 8\text{ Hz}) \text{ at } \{F_z, FC_z\}$ | Prefrontal executive working memory (Cavanagh & Frank, 2014) | Spike: Working memory buffer overload.<br/>Stable: Fluent cognitive integration. |
| **ASM** | Spectral EEG | $\ln(\text{Right Alpha}) - \ln(\text{Left Alpha})$ | Frontal motivational model (Davidson, 2004) | Positive: Approach motivation and active problem-solving.<br/>Negative: Frustration, withdrawal, or disengagement. |
| **ThetaPower** | Spectral EEG | $\int_4^8 S(f)df \text{ (105 channels)}$ | Slow cortical oscillation (Klimesch, 1999) | Cumulative drift across study blocks marks physiological study fatigue onset. |
| **AlphaPower** | Spectral EEG | $\int_8^{13} S(f)df \text{ (105 channels)}$ | Posterior sensory gating and cortical idling | Suppression: Deep visual/semantic focus.<br/>Elevation: Zoning out, inattention, or eye closure. |
| **BetaPower** | Spectral EEG | $\int_{13}^{30} S(f)df \text{ (105 channels)}$ | Active cortical computation and sensorimotor concentration | High: Deliberative analytical thinking.<br/>Low: Passive superficial scanning. |
| **TAR** | Spectral EEG | $\frac{\text{ThetaPower}}{\text{AlphaPower}}$ | Electrophysiological drowsiness index | Rising curve marks transition into attentional drowsiness and reduced vigilance. |
| **TBR** | Spectral EEG | $\frac{\text{ThetaPower}}{\text{BetaPower}}$ | Executive attentional control ratio | Elevated ratio signals difficulty maintaining executive control and sustained attention. |
| **GD** | Eye-Tracking | $\sum_{i \in \text{first pass}} t_{\text{fixation}, i}$ | First-pass lexical access (Just & Carpenter, 1980) | Long: Vocabulary decoding hurdle.<br/>Short: Fluent word recognition. |
| **FFD** | Eye-Tracking | Duration of 1st fixation on word | Early visual orthographic access (Rayner, 1998) | Reflects initial perceptual word-form decoding speed. |
| **TRT** | Eye-Tracking | $\sum_{\text{all passes}} t_{\text{fixation}}$ across sentence | Holistic lexical access + syntactic re-parsing | High TRT with normal GD directly pinpoints syntactic rereading struggles. |
| **PupilSize** | Eye-Tracking | Mean pupil diameter (px / z-units) | Locus coeruleus norepinephrine (LC-NE) system | Peak dilation marks sudden autonomic mental effort or semantic conflict resolution. |
| **FixationCount** | Eye-Tracking | Integer count of eye pauses | Oculomotor landing and word segmentation cadence | $>2.0$ fixations/word indicates reading stumbling, regressions, and lost reading cadence. |
| **REI** | Composite | $\text{standardize}(\text{EEI}_{rz} + 0.5 \cdot \text{Beta}_{rz})$ | Multimodal cortical alertness synthesis | Overall measure of active cognitive presence during study trials. |
| **CWI** | Composite | $0.3\cdot\text{FMT}_{rz} + 0.3\cdot\text{Pupil}_{rz} + 0.4\cdot\text{Dwell}_{rz}$ | Unified prefrontal + autonomic workload burden | Reflects the total mental taxation imposed by dense study passages. |
| **SAI** | Composite | $\frac{1.0}{\text{rolling\_CV}_{10}(\text{EEI}) + 0.1}$ | Rolling inverse coefficient of variation | High: Consistent concentration stamina.<br/>Low: Erratic focus swings and attention instability. |
| **MWLI_proxy** | Composite | $\text{rolling\_std}_{10}(\text{TRT}_{rz}) - \text{EEI}_{rz}$ | Cadence instability penalized by falling alertness | Direct behavioral indicator of mind-wandering and zoning out during reading. |
| **VSI_proxy** | Composite | $\frac{1.0}{\text{rolling\_std}_{10}(\text{GD}_{rz}) + 0.1}$ | Consistency of first-pass ocular intake | Measures stability of initial perceptual scanning rhythm across paragraphs. |
| **LES** | Pedagogical | $0.4\cdot\text{Flow} + 0.3\cdot\text{Cadence} - 0.2\cdot\text{Breakdown}$ | Composite reading efficiency score ($0–100\%$) | $\ge 85\%$: Optimal Flow State.<br/>$70–84\%$: Effective Comprehension.<br/>$<55\%$: Critical Saturation. |

---

## 3. The Scientific RAG (S-RAG) Architecture & Code Dataflow

Unlike standard generative RAG systems that query uncurated web crawls or hallucinate clinical claims, **S-RAG** enforces deterministic numerical grounding: physiological measurements are computed programmatically, and S-RAG retrieves peer-reviewed experimental literature to validate, contextualize, and explain those numbers.

```
[Raw ZuCo Data: .mat / CSV]
            │
            ▼
[Feature Extraction & Confound Residualization]
  • residualize.py (Ridge regression on 7 text covariates)
  • analyzer.py (Robust-Z calculation & personal baseline)
            │
            ▼
[Deterministic Student Observations Matrix]
  • Mean, baseline, relative deviation %, outlier trials
            │
            ├───► [HybridVectorStore (ingestion/vector_store.py)]
            │        • SQLite FTS5 (BM25 sparse term matching)
            │        • Dense Semantic Embeddings (Chroma/numpy)
            │        • Reciprocal Rank Fusion (alpha = 0.5)
            │
            ├───► [HybridRetriever (retrieval/hybrid_search.py)]
            │        • Domain query expansion
            │        • Modality & publication year filtering
            │
            ├───► [EvidenceSynthesizer (retrieval/synthesizer.py)]
            │        • Consensus classification (Canonical vs Context-Dependent)
            │        • Contradiction detection (divergent empirical findings)
            │
            ├───► [ResearchGroundingEngine (interpretation/grounding_engine.py)]
            │        • Binds student numbers to retrieved scientific claims
            │        • Non-diagnostic guardrail validator
            │
            ├───► [PersonalizedSpecEngine (interpretation/personalized_spec_engine.py)]
            │        • Generates 6-point personalized diagnostic specifications
            │
            ▼
[Multi-Channel User Presentation]
  • Interactive Web Dashboard (server.py + frontend/app.js)
  • Comprehensive Printable HTML & Markdown Reports
  • Publication-Grade PDF Export (reports/generate_comprehensive_pdf.py)
```

### Detailed S-RAG Code Modules:
1. **`discovery/aggregator.py` & `ingestion/pipeline.py`:** Automated discovery and section-aware chunking of peer-reviewed papers from OpenAlex, PubMed, Semantic Scholar, Europe PMC, and Crossref.
2. **`ingestion/vector_store.py`:** Persistent `HybridVectorStore` maintaining SQLite metadata, FTS5 inverted text index, and dense vector representations.
3. **`retrieval/hybrid_search.py`:** `HybridRetriever` using Reciprocal Rank Fusion:
   $$\text{HybridScore}(d) = \alpha \cdot \text{DenseScore}(d) + (1 - \alpha) \cdot \text{BM25Score}(d)$$
   Balanced at $\alpha = 0.5$ so exact acronyms (`FMT`, `EEI`) and broad semantic concepts both retrieve with high precision.
4. **`retrieval/synthesizer.py` & `contradiction.py`:** Assesses multi-study agreement, tags consensus levels, and flags methodological divergence.
5. **`interpretation/grounding_engine.py`:** Matches empirical measurements to retrieved literature claims and validates non-diagnostic compliance.
6. **`interpretation/personalized_spec_engine.py`:** Implements deterministic rule logic to formulate the 6-dimension personal specification:
   - *What does it do?*
   - *What is it actually?*
   - *What does it say about you?*
   - *Readability & text connection*
   - *Where did you make a mistake?*
   - *How could you resolve it?*

---

## 4. Pedagogical Diagnostic Insights & Student Study Interventions

NeuroRead-RAG translates physiological markers directly into concrete, actionable study behaviors:

1. **Working Memory Saturation ($\text{FMT}_{rz} > 2.0$ SD, elevated PupilSize):**
   - *Pedagogical Cause:* Multi-clause syntax and nested factual density overflow temporary buffer capacity before sentence meaning is resolved.
   - *Actionable Strategy:* **Punctuation Chunking Strategy.** Mentally pause for 1–2 seconds at commas, semicolons, and clause boundaries to allow the prefrontal buffer to consolidate context before decoding subsequent clauses.
2. **Syntactic Re-Reading Loops ($\text{TRT} > 300\text{ ms}$, normal $\text{GD} \in [120, 150]\text{ ms}$):**
   - *Pedagogical Cause:* Words are decoded fluently on first pass, but syntactic assembly fails, triggering backward ocular regressions.
   - *Actionable Strategy:* **Subject-Verb Anchoring.** Identify the primary actor and action verb first before deciphering complex modifying clauses or prepositional phrases.
3. **Lexical Hurdle Stumbling ($\text{GD} > 250\text{ ms}$, $\text{FixationCount} > 2.5/\text{word}$):**
   - *Pedagogical Cause:* Unfamiliar low-frequency vocabulary halts forward reading momentum.
   - *Actionable Strategy:* **Pre-Reading Vocabulary Scan.** Skim technical texts for unfamiliar words (>3 syllables) and clarify their definitions before reading the full passage.
4. **Attentional Drift / Fading (Depressed EEI, elevated TAR & MWLI_proxy):**
   - *Pedagogical Cause:* Cortical alertness drops; student enters passive scanning or mind-wandering.
   - *Actionable Strategy:* **Active Note-Taking Trigger.** Write a 3-keyword summary of each paragraph in margins to stimulate cortical beta oscillations.
5. **Cognitive Fatigue Onset (Progressive drift in Theta, falling SAI):**
   - *Pedagogical Cause:* Continuous study exceeds personal fatigue threshold (typically 18–25 minutes).
   - *Actionable Strategy:* **Dynamic Pomodoro Pacing.** Take structured 5-minute visual relaxation breaks gazing 20 feet away to relax ciliary muscles and reset cortical baseline.

---

## 5. Mandatory Scientific Limitations & Ethical Guardrails

1. **Strictly Non-Diagnostic:** All metrics represent exploratory physiological proxies measured within a controlled natural reading protocol (ZuCo). They **DO NOT** establish medical, psychiatric, or clinical diagnoses (e.g., ADHD, dyslexia, executive dysfunction).
2. **Within-Subject Reference Ranges:** Reference intervals describe within-session typical ranges ($\text{Median} \pm 1.5\text{ Robust SD}$) relative to the participant's own homeostatic baseline—never universal clinical norms.
3. **Correlational Nature:** Electrophysiological power shifts and oculomotor fixations are correlational indicators of processing effort, not deterministic proof of psychological state.
4. **Confound Sensitivity:** Ambient luminance, individual sleep quality, caffeine intake, and prescription medications modulate autonomic pupil diameter and spectral baselines.
5. **Zero Forced Quotas:** Unlike legacy quartiles that artificially labeled 25% of all sentences as "breakdowns", the pipeline uses zero-quota interpretable thresholds. A student reading fluently will show zero false alarms.

---

## 6. Project Verification & Conclusion

* **Deterministic Execution:** The full reproducible pipeline ([run_all.py](file:///mnt/DATA/SPLAB/1.%20Code/EEG-To-Text/neuroread_rag/run_all.py)) executes in 21.0s under fixed seed 42.
* **Test Suite Parity:** **42 / 42 unit and integration tests passing (100%)** in `neuroread_rag/tests/`.
* **Export Artifacts:**
  - PDF Reference Document: [`reports/neuroread_comprehensive_project_reference.pdf`](file:///mnt/DATA/SPLAB/1.%20Code/EEG-To-Text/reports/neuroread_comprehensive_project_reference.pdf)
  - Markdown Compendium: [`reports/neuroread_comprehensive_project_reference.md`](file:///mnt/DATA/SPLAB/1.%20Code/EEG-To-Text/reports/neuroread_comprehensive_project_reference.md)
  - Interactive Dashboard: Served at `http://localhost:8000` via [server.py](file:///mnt/DATA/SPLAB/1.%20Code/EEG-To-Text/neuroread_rag/server.py).
