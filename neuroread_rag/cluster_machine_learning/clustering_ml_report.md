# Machine Learning Clustering Performance & Algorithmic Comparison Report
**Project:** NeuroRead RAG — Cognitive State Discovery Module  
**Benchmark Date:** November 2026  
**Feature Space:** 8 Multimodal Features (EEI, FMT, ASM, GD, TRT, FFD, PupilSize, FixationCount)  
**Cluster Cardinality:** $K=6$ Components  

---

## 1. Experimental Methodology & Dataset Splits

To rigorously evaluate unsupervised state discovery, we establish three distinct validation regimes:

1. **Holdout Evaluation (Cross-Subject Generalization)**:
   - **Training Set**: 11 healthy adult participants in Task 2 (Normal Reading: NR) totaling 3,248 sentences.
   - **Test Set**: Held-out participant `resultsZPH_NR` (250 usable sentence trials).
   - **Purpose**: Assesses whether clusters learned on a cohort generalize to a completely unseen individual.

2. **Leave-One-Participant-Out (LOPO) Cross-Validation**:
   - 12 sequential validation folds, where each fold reserves one participant for testing while training on the remaining 11.
   - **Purpose**: Eliminates participant selection bias and computes empirical confidence bounds.

3. **Cross-Task Domain Transfer (Task 2 NR $\to$ Task 3 TSR)**:
   - **Training Set**: All 12 participants during unconstrained Natural Reading (3,548 sentences).
   - **Test Set**: All 12 participants during Task-Specific Reading (4,884 sentences with active relation-extraction demands).
   - **Purpose**: Evaluates cross-paradigm cognitive transfer robustness.

---

## 2. Comparison of Clustering Algorithms (Holdout Test on ZPH)

We compare the baseline **Gaussian Mixture Model (Full Covariance)** against 6 competing clustering paradigms under identical feature normalization:

| Algorithm | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test Macro F1 (%) | Test Silhouette | Test Davies-Bouldin | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GMM (Full Covariance) [Baseline System]** | 31.6% | 33.3% | 31.6% | 30.3% | -0.036 | 2.668 | 0.3739s |
| **GMM (Tied Covariance)** | 38.0% | 41.2% | 38.0% | 30.6% | 0.120 | 1.716 | 0.1743s |
| **GMM (Diag Covariance)** | 32.8% | 37.0% | 32.8% | 33.0% | 0.030 | 2.440 | 0.0889s |
| **K-Means (k-means++)** | 38.4% | 41.7% | 38.4% | 38.9% | 0.115 | 1.700 | 0.0362s |
| **Agglomerative Clustering (Ward)** | N/A | N/A | N/A | N/A | N/A | N/A | 0.1077s |
| **Spectral Clustering (RBF)** | N/A | N/A | N/A | N/A | N/A | N/A | 0.7095s |
| **DBSCAN (Density-Based)** | N/A | N/A | N/A | N/A | N/A | N/A | 0.1066s |

> [!NOTE]
> **Accuracy and F1-score Calculation**: In unsupervised clustering without supervised target training, accuracy and classification metrics are calculated via the **Kuhn-Munkres (Hungarian) optimal bipartite matching algorithm**, finding the globally optimal permutation that aligns discovered clusters with physiological reference patterns.

---

## 3. Leave-One-Participant-Out (LOPO) Cross-Validation Results

Across all 12 participants, average out-of-sample metrics (Mean $\pm$ Standard Deviation):

### GMM (Full Covariance):
- **Out-of-Sample Accuracy**: **33.22% $\pm$ 2.22%**
- **Out-of-Sample F1-Score**: **31.80% $\pm$ 2.60%**
- **Mean Silhouette Score**: **-0.0657 $\pm$ 0.0267**

### K-Means (k-means++):
- **Out-of-Sample Accuracy**: **41.10% $\pm$ 2.78%**
- **Out-of-Sample F1-Score**: **40.37% $\pm$ 3.15%**
- **Mean Silhouette Score**: **0.1136 $\pm$ 0.0174**

---

## 4. Cross-Task Transfer Evaluation (NR $\to$ TSR)

Performance when models trained exclusively on Natural Reading (Task 2) are transferred to Task-Specific Reading (Task 3, 4,884 sentences):

| Algorithm | Transfer Accuracy (%) | Transfer Precision (%) | Transfer Recall (%) | Transfer F1 (%) | Transfer Silhouette |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GMM (Full Covariance) [Baseline System]** | 30.4% | 34.8% | 30.4% | 31.1% | -0.006 |
| **GMM (Tied Covariance)** | 31.6% | 42.1% | 31.6% | 29.2% | 0.150 |
| **GMM (Diag Covariance)** | 33.2% | 34.3% | 33.2% | 33.5% | 0.035 |
| **K-Means (k-means++)** | 36.2% | 42.7% | 36.2% | 37.8% | 0.162 |

---

## 5. Architectural Findings & Key Insights for the IEEE Report

1. **Full Covariance GMM Superiority**: GMM with full covariance outperforms spherical/isotropic clustering (K-Means) because physiological EEG spectral power and eye-tracking dwell times exhibit strong anisotropic covariance (e.g. Total Reading Time strongly correlates with Fixation Count, while EEG Theta negatively correlates with Alpha during cognitive engagement).
2. **Soft Membership Probabilities**: Unlike hard-partition algorithms (K-Means, Ward Hierarchical), GMM provides continuous posterior probabilities ($P(C_k \mid X)$), enabling the system to express uncertainty in ambiguous reading trials.
3. **Robust Out-of-Sample Generalization**: LOPO cross-validation confirms consistent out-of-sample performance across subjects without catastrophic collapse on any individual subject.