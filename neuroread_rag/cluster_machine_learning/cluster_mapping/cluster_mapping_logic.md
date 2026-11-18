# NeuroRead-RAG: Cognitive Cluster Mapping & Physiological Separation Logic

---

## 1. Overview & Architectural Purpose

Unsupervised clustering algorithms (such as GMM, K-Means, Fuzzy C-Means, HDBSCAN, and Agglomerative Hierarchical Clustering) assign data points to arbitrary numerical identifiers ($0, 1, 2, \dots, K-1$). These integers have no intrinsic physiological meaning: Cluster 0 in one run could correspond to cognitive fatigue, while in another run it could correspond to fluent reading.

To convert raw statistical partitions into meaningful, clinically grounded pedagogical diagnostics, the **Cluster Mapping Engine** inspects the **centroid feature vector** of each discovered cluster across the 10 multimodal brain and eye-tracking features. It projects these centroids onto neurophysiological profiles established in reading cognitive neuroscience, assigning each cluster an objective cognitive state name and category:
- **Concentrated Reading States** (Concentrated Fluent Reading, Deep Cognitive Processing)
- **Fatigued State** (Cognitive Fatigue / Low Vigilance)
- **Workload / Bottleneck States** (High Cognitive Workload, Comprehension Bottleneck)
- **Outlier Noise** (Atypical physiological artifacts)

This document details the exact mathematical formulas, feature thresholds, and decision logic used to separate and name each cluster.

---

## 2. Physiological Feature Vector Representation

Clustering is performed in a 10-dimensional standardized feature space ($Z$-scores relative to cohort baselines, computed via `RobustScaler`):

$$\mathbf{z}_i = \left[ z_{\text{EEI}}, z_{\text{FMT}}, z_{\text{ASM}}, z_{\text{GD}}, z_{\text{TRT}}, z_{\text{FFD}}, z_{\text{Pupil}}, z_{\text{Fix}}, z_{\text{Alpha}}, z_{\text{Beta}} \right]^\top$$

Where:
1. **$z_{\text{EEI}}$ (Engagement / Executive Index)**: $\beta / (\alpha + \theta)$. Cortical activation and executive processing during reading. Positive values indicate active focus; negative values indicate attentional disengagement.
2. **$z_{\text{FMT}}$ (Frontal Midline Theta, 4–8 Hz)**: Anterior cingulate cortex activation reflecting working memory maintenance, cognitive effort, and task difficulty.
3. **$z_{\text{ASM}}$ (Alpha Spectral Asymmetry, 8–13 Hz)**: Frontal hemispheric power balance reflecting motivational approach vs. avoidance behavior.
4. **$z_{\text{Alpha}}$ (Parietal-Occipital Alpha, 8–13 Hz)**: Inverse marker of cortical arousal. Substantially elevated during cognitive fatigue, drowsiness, and mind-wandering.
5. **$z_{\text{Beta}}$ (Frontal-Central Beta, 13–30 Hz)**: Reflects active mental workload, alertness, and motor/attentional readiness.
6. **$z_{\text{GD}}$ (Gaze Duration, ms)**: Total duration of fixations on a word before moving away; reflects lexical access and semantic lookup.
7. **$z_{\text{TRT}}$ (Total Reading Time, ms)**: Total time spent reading a word including late regressions; measures comprehension difficulty and syntactic reanalysis.
8. **$z_{\text{FFD}}$ (First Fixation Duration, ms)**: Duration of the very first gaze fixation on a word; reflects early visual/orthographic processing.
9. **$z_{\text{Pupil}}$ (Mean Pupil Diameter)**: Autonomic arousal index driven by the locus coeruleus-norepinephrine system; dilates during high mental effort and constricts during fatigue.
10. **$z_{\text{Fix}}$ (Total Fixations)**: Number of discrete gaze pauses per sentence; elevated fixation counts reflect reading disfluency.

---

## 3. Mathematical Affinity Scoring Formulas

For each discovered cluster $k$, let its centroid vector in the standardized feature space be:
$$\mathbf{c}_k = \left[ c_{k,\text{EEI}}, c_{k,\text{FMT}}, c_{k,\text{ASM}}, c_{k,\text{GD}}, c_{k,\text{TRT}}, c_{k,\text{FFD}}, c_{k,\text{Pupil}}, c_{k,\text{Fix}}, c_{k,\text{Alpha}}, c_{k,\text{Beta}} \right]$$

The mapping engine evaluates 5 linear projection affinity scores $A(k, S)$ for each candidate cognitive state $S$:

### State 1: Concentrated Fluent Reading (Category: Concentrated)
$$A(k, \text{Fluent}) = 1.5 \, c_{k,\text{EEI}} - 0.8 \, c_{k,\text{TRT}} - 0.8 \, c_{k,\text{GD}} - 0.5 \, c_{k,\text{FFD}} - 0.5 \, c_{k,\text{Alpha}} - 0.5 \, c_{k,\text{Fix}}$$
- **Separation Logic**: Demands high cortical engagement ($c_{\text{EEI}} > 0$) combined with swift, efficient eye movements (negative $Z$-scores for GD, TRT, FFD, and FixationCount).
- **Behavioral Profile**: The student reads fluently and effortlessly; words are recognized instantly on first pass without need for rereading or cognitive strain.

### State 2: Deep Cognitive Processing (Category: Concentrated)
$$A(k, \text{Deep}) = 1.2 \, c_{k,\text{EEI}} + 0.8 \, c_{k,\text{TRT}} + 0.8 \, c_{k,\text{GD}} + 0.5 \, c_{k,\text{FMT}} - 0.3 \, c_{k,\text{Alpha}}$$
- **Separation Logic**: Characterized by high executive engagement ($c_{\text{EEI}} > 0$) paired with **deliberate, prolonged reading times** ($c_{\text{TRT}} > 0, c_{\text{GD}} > 0$) and moderate theta activation ($c_{\text{FMT}} > 0$).
- **Behavioral Profile**: The student is deeply focused on semantically rich, syntactically complex sentences; gaze durations are long not because of confusion, but due to deep conceptual integration.

### State 3: Cognitive Fatigue / Low Vigilance (Category: Fatigued)
$$A(k, \text{Fatigue}) = -1.5 \, c_{k,\text{EEI}} + 1.0 \, c_{k,\text{Alpha}} + 0.8 \, c_{k,\text{FMT}} - 0.8 \, c_{k,\text{Pupil}}$$
- **Separation Logic**: Defined by the classic neurophysiological triad of mental exhaustion:
  1. Depressed executive index ($c_{\text{EEI}} \ll 0$)
  2. Elevated synchronized alpha and theta power ($c_{\text{Alpha}} > 0, c_{\text{FMT}} > 0$) reflecting cortical slowing
  3. Autonomic pupil constriction ($c_{\text{Pupil}} < 0$) indicating reduced sympathetic arousal
- **Behavioral Profile**: The student is mentally depleted, experiencing lapses of sustained attention and drowsiness. Reading pace slows without corresponding semantic encoding.

### State 4: High Cognitive Workload (Category: Workload)
$$A(k, \text{Workload}) = 1.5 \, c_{k,\text{FMT}} + 1.2 \, c_{k,\text{Pupil}} + 1.0 \, c_{k,\text{Beta}} + 0.5 \, c_{k,\text{Alpha}}$$
- **Separation Logic**: Triggered by intense frontal midline theta power ($c_{\text{FMT}} \gg 0$, reflecting anterior cingulate cortex load) coupled with pronounced autonomic pupil dilation ($c_{\text{Pupil}} > 0$) and elevated beta power ($c_{\text{Beta}} > 0$).
- **Behavioral Profile**: The student is exerting maximal mental effort to maintain multi-clause sentence structures in working memory; high cognitive strain.

### State 5: Comprehension Bottleneck (Category: Bottleneck)
$$A(k, \text{Bottleneck}) = 1.5 \, c_{k,\text{TRT}} + 1.3 \, c_{k,\text{Fix}} + 1.0 \, c_{k,\text{GD}} - 0.5 \, c_{k,\text{EEI}}$$
- **Separation Logic**: Dominated by extreme reading disfluency: total reading time is heavily extended ($c_{\text{TRT}} \gg 0$), fixation count is very high ($c_{\text{Fix}} \gg 0$), and gaze durations are prolonged, while EEI remains neutral or negative.
- **Behavioral Profile**: The student is struggling with unfamiliar vocabulary or ambiguous syntax, executing multiple regressions and refixations to decode the text.

---

## 4. Bipartite Optimal Assignment Algorithm

To ensure each of the $K=5$ clusters receives a unique, non-overlapping cognitive state name:
1. Construct the $5 \times 5$ affinity matrix $\mathbf{A} \in \mathbb{R}^{5 \times 5}$ where $A_{k, s} = A(k, S_s)$.
2. Flatten all pairs $(k, s)$ and sort by affinity score in descending order.
3. Greedily match cluster $k$ with cognitive state $s$: once a cluster or state is assigned, it cannot be reused.
4. If density-based methods (such as HDBSCAN) produce an outlier label ($-1$), it is automatically mapped to:
   - **Name**: `Unclassified Outlier / Noise`
   - **Category**: `Noise`
   - **Description**: Physiological readings that do not conform to any coherent reading cluster.

---

## 5. Aggregation into Student Diagnostic Percentages

Once every sentence $i \in \{1, \dots, N\}$ is assigned a cluster label $k_i$, the student dashboard computes the percentage of sentences in each state:

$$P(\text{State}_s) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{Cluster}(i) \mapsto \text{State}_s \right) \times 100\%$$

The broad diagnostic categories are defined as:
- **Total Concentrated Reading (%)**:
  $$\text{Concentrated} \% = P(\text{Concentrated Fluent Reading}) + P(\text{Deep Cognitive Processing})$$
- **Total Cognitive Fatigue (%)**:
  $$\text{Fatigued} \% = P(\text{Cognitive Fatigue / Low Vigilance})$$
- **Total Workload / Bottleneck (%)**:
  $$\text{Workload / Bottleneck} \% = P(\text{High Cognitive Workload}) + P(\text{Comprehension Bottleneck})$$

This percentage breakdown forms the direct input to the Scientific RAG (S-RAG) recommendation engine, triggering micro-breaks when fatigue exceeds 40% or sentence simplification when bottlenecks exceed 15%.
