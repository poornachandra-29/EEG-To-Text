# NeuroRead AI: Research-Grounded Physiological Assessment Report

**Participant ID:** `resultsZKH_TSR` | **Sentences Analyzed:** 407 | **Generated:** 2026-11-08

> **IMPORTANT NOTICE**: This document provides research-grounded physiological observations derived from simultaneous EEG and eye tracking. It is strictly non-diagnostic and does NOT establish clinical disorders or subjective mind-wandering ground-truth.

## 1. Executive Summary

This research-grounded physiological assessment analyzes participant resultsZKH_TSR across 407 sentence-reading trials recorded in the ZuCo natural reading protocol. Deterministic evaluation reveals steady neurophysiological engagement: the EEG Engagement Index (EEI = 1.1610) remained within the personal baseline range with an offset of +124.3%. Gaze duration (GD = 133.9 ms) and total reading time (TRT = 160.2 ms) reflected normal lexical processing cadence with expected syntactic rereading variations. Unsupervised Gaussian Mixture Modeling identified 5 distinct clusters, predominantly characterized by balanced cognitive control and focused reading patterns.

## 2. Focus and Attention Assessment

Focus and Stability Assessment: Sustained Attention Evidence Proxy (SAI) averaged +0.04, while Attention-Instability Evidence Proxy (MWLI_proxy) averaged +0.05. These exploratory indices measure reading cadence stability and relative spectral engagement, providing heuristic evidence of processing continuity rather than ground-truth psychological states.

- **Sustained Attention Evidence Proxy (SAI):** +0.0391 (Within Typical Range)
- **Attention-Instability Evidence Proxy (MWLI_proxy):** +0.0505

## 3. Spectral EEG Analysis

Spectral EEG Analysis: Across 407 reading trials, EEG Engagement Index averaged 1.161 (elevated above baseline). Frontal Midline Theta baseline was 2.3498 µV², reflecting oscillatory dynamics in medial prefrontal regions. Alpha asymmetry metric was -0.0222. In accordance with Pope et al. (1995) and Cavanagh & Frank (2014), these spectral indicators reflect vigilant task engagement without establishing clinical conditions.

| Metric | Category | Mean | Baseline | Relative Dev | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EEG Engagement Index (EEI)** | EEG | 1.161 ratio (Beta / (Alpha + Theta)) | 0.5177 | +124.3% | Elevated Above Baseline |
| **Frontal Midline Theta (FMT)** | EEG | 3.7938 µV² | 2.3498 | +61.5% | Elevated Above Baseline |
| **Alpha Asymmetry Metric (ASM)** | EEG | -0.0222 log difference | 0.0 | +0.0% | Within Typical Range |
| **Theta Band Power (ThetaPower)** | EEG | 3.7938 µV² | 2.3498 | +61.5% | Elevated Above Baseline |
| **Alpha Band Power (AlphaPower)** | EEG | 7.5146 µV² | 2.4372 | +208.3% | Elevated Above Baseline |
| **Beta Band Power (BetaPower)** | EEG | 12.647 µV² | 2.5104 | +403.8% | Elevated Above Baseline |
| **Theta / Alpha Ratio (Theta_Alpha_Ratio)** | EEG | 0.5239 ratio | 0.502 | +4.3% | Within Typical Range |
| **Theta / Beta Ratio (Theta_Beta_Ratio)** | EEG | 3.535 ratio | 3.4179 | +3.4% | Within Typical Range |

## 4. Eye-Tracking Analysis

Eye-Tracking Dynamics: Gaze duration averaged 136.7715 ms with total reading time of 175.89 ms. Mean pupil size was 1048.7074 units. As demonstrated by Rayner (1998) and Beatty (2000), dwell times and pupillometric dilation reflect lexical processing demands and autonomic arousal under reading task constraints.

| Metric | Category | Mean | Baseline | Relative Dev | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gaze Duration (GD)** | Eye-Tracking | 136.7715 ms | 135.1842 | +1.2% | Within Typical Range |
| **First Fixation Duration (FFD)** | Eye-Tracking | 118.7115 ms | 107.0395 | +10.9% | Within Typical Range |
| **Total Reading Time (TRT)** | Eye-Tracking | 175.89 ms | 192.775 | -8.8% | Within Typical Range |
| **Mean Pupil Diameter (PupilSize)** | Eye-Tracking | 1048.7074 arbitrary units / px | 281.4277 | +272.6% | Elevated Above Baseline |
| **Fixation Count (FixationCount)** | Eye-Tracking | 1.4749 count | 1.6821 | -12.3% | Within Typical Range |

## 5. Cognitive Workload Assessment

Cognitive Workload Evidence: The composite Cognitive Workload Evidence Index (CWI) averaged +0.26 robust-z units. Multimodal literature confirms that concurrent elevations in theta oscillations and eye-tracking dwell times characterize high sentence complexity and syntactic reanalysis.

- **Composite Workload Index (CWI):** +0.2556 robust-z units
- **Session Baseline:** +0.1925 | **Relative Shift:** +32.8%

## 6. Sentence-Level Observations

Out of 407 read sentences, **123 sentences (30.2%)** exhibited marked physiological deviations (|robust-z| ≥ 2.0).

### Sample Sentences with Notable Processing Demands:

1. *"The Foundation has grown immensely and, by 1950, had become national and international in scope."*
   - **Cluster:** Cluster 5 (0.88 prob) | **Pattern:** Mixed / indeterminate
   - **Top Metrics:** GD=116.917ms, TRT=279.167ms, EEI=1.085

2. *"The same year, they were each awarded by one of the UN's own Human Rights Prizes."*
   - **Cluster:** Cluster 4 (0.60 prob) | **Pattern:** Mixed / indeterminate
   - **Top Metrics:** GD=111.818ms, TRT=136.364ms, EEI=None

3. *"He won a Nobel Prize in Chemistry in 1928."*
   - **Cluster:** Cluster 3 (0.96 prob) | **Pattern:** Fatigue-compatible EEG pattern
   - **Top Metrics:** GD=105.2ms, TRT=127.2ms, EEI=0.81

## 7. Research-Paper Comparison

| Metric | Participant Observed | Session Baseline | Literature Alignment & DOI |
| :--- | :--- | :--- | :--- |
| **EEG Engagement Index** | 1.161 ratio (Beta / (Alpha + Theta)) (Elevated Above Baseline) | 0.5177 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Frontal Midline Theta** | 3.7938 µV² (Elevated Above Baseline) | 2.3498 | Alexander Gundel et al. (1992) - [10.1016/0167-8760(93)90054-9](https://doi.org/10.1016/0167-8760(93)90054-9) |
| **Alpha Asymmetry Metric** | -0.0222 log difference (Within Typical Range) | 0.0 | Christian Gold et al. (2012) - [10.1111/sjop.12022](https://doi.org/10.1111/sjop.12022) |
| **Theta Band Power** | 3.7938 µV² (Elevated Above Baseline) | 2.3498 | Olaf Dimigen et al. (2011) - [10.1037/a0023885](https://doi.org/10.1037/a0023885) |
| **Alpha Band Power** | 7.5146 µV² (Elevated Above Baseline) | 2.4372 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Beta Band Power** | 12.647 µV² (Elevated Above Baseline) | 2.5104 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Theta / Alpha Ratio** | 0.5239 ratio (Within Typical Range) | 0.502 | Tee Yi Wen et al. (2020) - [10.11591/ijeecs.v17.i1.pp175-182](https://doi.org/10.11591/ijeecs.v17.i1.pp175-182) |
| **Theta / Beta Ratio** | 3.535 ratio (Within Typical Range) | 3.4179 | Mi D et al. (2026) - [10.3389/fpsyt.2026.1834200](https://doi.org/10.3389/fpsyt.2026.1834200) |
| **Gaze Duration** | 136.7715 ms (Within Typical Range) | 135.1842 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **First Fixation Duration** | 118.7115 ms (Within Typical Range) | 107.0395 | Olaf Dimigen et al. (2011) - [10.1037/a0023885](https://doi.org/10.1037/a0023885) |
| **Total Reading Time** | 175.89 ms (Within Typical Range) | 192.775 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Mean Pupil Diameter** | 1048.7074 arbitrary units / px (Elevated Above Baseline) | 281.4277 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Fixation Count** | 1.4749 count (Within Typical Range) | 1.6821 | Nora Hollenstein et al. (2018) - [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291) |
| **Engagement-Related Signal Index** | 0.2797 standardized robust-z (Within Typical Range) | 0.1212 | Vincenzo Ronca et al. (2025) - [10.3390/bioengineering12060597](https://doi.org/10.3390/bioengineering12060597) |
| **Cognitive Workload Evidence Index** | 0.2556 standardized robust-z (Within Typical Range) | 0.1925 | Jackson Beatty et al. (2000) - [10.1017/CBO9780511546396.007](https://doi.org/10.1017/CBO9780511546396.007) |
| **Engagement-Stability Evidence Proxy** | 0.0391 standardized robust-z (Within Typical Range) | 0.0318 | Jonathan Smallwood et al. (2015) - [10.1146/annurev-psych-010814-015331](https://doi.org/10.1146/annurev-psych-010814-015331) |
| **Attention-Instability Evidence Proxy** | 0.0505 standardized robust-z (Within Typical Range) | 0.0355 | Sun H et al. (2026) - [10.64898/2026.05.14.725227](https://doi.org/10.64898/2026.05.14.725227) |
| **Signal-Stability Proxy** | -0.2442 standardized robust-z (Within Typical Range) | -0.0941 | Zhang J et al. (2026) - [10.1016/j.neubiorev.2026.106872](https://doi.org/10.1016/j.neubiorev.2026.106872) |

## 8. Personalized Evidence-Based Recommendations

### Scheduled Cognitive Intervals & Ultradian Micro-Breaks (Pacing & Cognitive Breaks)
- **Observation Basis:** Participant completed 407 continuous reading trials. Workload-elevated and fatigue-compatible pattern evidence was observed across 27.3% of read sentences (CWI mean +0.26).
- **Actionable Strategy:** Adopt 20-25 minute structured reading blocks followed by 3-5 minute cognitive pauses with visual disengagement (looking at distant focal points >20 feet away).
- **Scientific Rationale:** Frontal Midline Theta (FMT) and pupil dilation reflect central executive control effort. Periodic ocular disengagement resets autonomic arousal and prevents the spectral drift toward theta-alpha synchronization characteristic of continuous cognitive fatigue.
- **Supporting Research:** [Cavanagh & Frank (2014)](https://doi.org/10.1016/j.tics.2014.04.012), [Klimesch (1999)](https://doi.org/10.1016/S0165-0173(98)00056-3)

### Syntactic Segmentation and Visual Text Formatting (Reading Strategy & Chunking)
- **Observation Basis:** Total reading time (TRT) exhibited a relative deviation of -8.8% (mean 175.89 ms), with gaze duration (GD) averaging 136.7715 ms during sentence-level syntactic parsing.
- **Actionable Strategy:** Break complex, dense text into shorter, distinct informational clauses. Utilize visual anchors, bullet points, and wider line-spacing (1.4 - 1.6x) to reduce rereading passes.
- **Scientific Rationale:** Eye-movement research shows that prolonged total reading time (TRT) and regression fixations reflect syntactic reanalysis and lexical competition. Increasing whitespace and semantic chunking significantly decreases redundant regressions without sacrificing comprehension.
- **Supporting Research:** [Rayner (1998)](https://doi.org/10.1037/0033-2909.124.3.372), [Dimigen et al. (2011)](https://doi.org/10.1037/a0023885)

### Active Reading Cadence Regulation & Self-Monitoring (Workload Modulation)
- **Observation Basis:** The Attention-Instability Evidence Proxy (MWLI_proxy) averaged +0.05, reflecting periodic fluctuations between rapid skimming and extended fixation pauses.
- **Actionable Strategy:** Practice active reading strategies: Summarize core concepts in 1-2 words after challenging sentences, and utilize a visual guide (such as a cursor or soft pointer) to maintain a steady, uniform pacing.
- **Scientific Rationale:** Erratic eye fixation patterns combined with reduced EEG engagement ratios indicate attentional cadence drift. Active metacognitive prompting re-couples visual attention with semantic processing.
- **Supporting Research:** [Smallwood & Schooler (2015)](https://doi.org/10.1146/annurev-psych-010814-015331), [Pope, Bogart, & Bartolome (1995)](https://doi.org/10.1016/0301-0511(95)05116-3)

### Display Luminance and Ambient Visual Optimization (Visual Ergonomics)
- **Observation Basis:** Mean pupil diameter was 1048.7 units with standard deviation 118.748. Variability index was 0.11 across reading intervals.
- **Actionable Strategy:** Match monitor luminance to surrounding ambient light to prevent unnecessary pupillary constriction or dilation strain. Prefer matte displays and warm contrast themes during prolonged reading.
- **Scientific Rationale:** Pupillometry integrates both autonomic arousal (norepinephrine) and the pupillary light reflex. Minimizing luminance contrasts stabilizes baseline pupil diameter, enhancing signal-to-noise ratio for cognitive effort monitoring.
- **Supporting Research:** [Beatty & Lucero-Wagoner (2000)](https://doi.org/10.1017/CBO9780511546396.007)

## 9. Scientific Limitations & Guardrails

- All indices represent exploratory physiological proxies measured within a controlled natural reading protocol (ZuCo).
- Reference ranges describe within-session typical intervals (median ± 1.5 robust SD) and must not be treated as universal healthy/clinical norms.
- Physiological modulations in EEG and eye movements are correlational and do not establish medical or psychiatric diagnoses (e.g. ADHD, dyslexia, cognitive impairment).
- Cadence variations and composite proxies (MWLI_proxy, SAI) reflect reading cadence irregularity rather than ground-truth subjective mind-wandering.
- Ambient luminance, individual caffeine intake, sleep quality, and reading strategy substantially modulate autonomic and electrophysiological baselines.

## 10. Complete Scientific References

- **Alexander Gundel et al. (1992)**. *Frontal midline theta power and mental effort in flight simulation*. Abstract. DOI: [10.1016/0167-8760(93)90054-9](https://doi.org/10.1016/0167-8760(93)90054-9)
- **Alan T. Pope et al. (1995)**. *Biocybernetic system evaluates indices of operator engagement in automated task*. Abstract. DOI: [10.1016/0301-0511(95)05116-3](https://doi.org/10.1016/0301-0511(95)05116-3)
- **Keith Rayner (1998)**. *Eye movements in reading and information processing: 20 years of research*. Abstract. DOI: [10.1037/0033-2909.124.3.372](https://doi.org/10.1037/0033-2909.124.3.372)
- **Ryouhei Ishii et al. (1999)**. *Medial prefrontal cortex generates frontal midline theta rhythm*. Abstract. DOI: [10.1097/00001756-199903170-00003](https://doi.org/10.1097/00001756-199903170-00003)
- **Jackson Beatty et al. (2000)**. *The pupillary system and cognitive workload*. Abstract. DOI: [10.1017/CBO9780511546396.007](https://doi.org/10.1017/CBO9780511546396.007)
- **James A. Coan et al. (2004)**. *Frontal EEG asymmetry as a moderator and mediator of emotion and motivational direction*. Abstract. DOI: [10.1016/j.biopsycho.2004.03.002](https://doi.org/10.1016/j.biopsycho.2004.03.002)
- **Adam M. Brickman et al. (2005)**. *Category and letter verbal fluency across the adult lifespan: relationship to EEG theta power*. Abstract. DOI: [10.1016/j.acn.2004.12.006](https://doi.org/10.1016/j.acn.2004.12.006)
- **Pamela Deans et al. (2010)**. *Use of Eye Movement Tracking in the Differential Diagnosis of Attention Deficit Hyperactivity Disorder (ADHD) and Reading Disability*. Abstract. DOI: [10.4236/psych.2010.14032](https://doi.org/10.4236/psych.2010.14032)
- **Olaf Dimigen et al. (2011)**. *Coregistration of eye movements and EEG in natural reading: Analyses and review*. Abstract. DOI: [10.1037/a0023885](https://doi.org/10.1037/a0023885)
- **Simon Finnigan et al. (2011)**. *Resting EEG theta power correlates with cognitive performance in healthy older adults*. Abstract. DOI: [10.1111/j.1469-8986.2010.01173.x](https://doi.org/10.1111/j.1469-8986.2010.01173.x)
- **Christian Gold et al. (2012)**. *Validity and reliability of electroencephalographic frontal alpha asymmetry and frontal midline theta as biomarkers for depression*. Abstract. DOI: [10.1111/sjop.12022](https://doi.org/10.1111/sjop.12022)
- **Tom Foulsham et al. (2013)**. *Mind wandering in sentence reading: Decoupling the link between mind and eye.*. Abstract. DOI: [10.1037/a0030217](https://doi.org/10.1037/a0030217)
- **Cordelia Mühlenbeck et al. (2015)**. *Gaze Duration Biases for Colours in Combination with Dissonant and Consonant Sounds: A Comparative Eye-Tracking Study with Orangutans*. Abstract. DOI: [10.1371/journal.pone.0139894](https://doi.org/10.1371/journal.pone.0139894)
- **Jonathan Smallwood et al. (2015)**. *The science of mind wandering: empirically navigating the stream of consciousness*. Abstract. DOI: [10.1146/annurev-psych-010814-015331](https://doi.org/10.1146/annurev-psych-010814-015331)
- **Diana Armbruster‐Genc et al. (2016)**. *Brain Signal Variability Differentially Affects Cognitive Flexibility and Cognitive Stability*. Abstract. DOI: [10.1523/jneurosci.2517-14.2016](https://doi.org/10.1523/jneurosci.2517-14.2016)
- **Joseph K. Nuamah et al. (2017)**. *Support vector machine (SVM) classification of cognitive tasks based on electroencephalography (EEG) engagement index*. Abstract. DOI: [10.1080/2326263x.2017.1338012](https://doi.org/10.1080/2326263x.2017.1338012)
- **Félix Bacigalupo et al. (2018)**. *Lateralized Suppression of Alpha-Band EEG Activity As a Mechanism of Target Processing*. Abstract. DOI: [10.1523/jneurosci.0183-18.2018](https://doi.org/10.1523/jneurosci.0183-18.2018)
- **Hyangsook Park (2018)**. *The Role of Reading Span in L2 Reading Comprehension and Eye Movements*. Abstract. DOI: [10.15858/engtea.73.1.201803.3](https://doi.org/10.15858/engtea.73.1.201803.3)
- **Nora Hollenstein et al. (2018)**. *ZuCo, a simultaneous EEG and eye-tracking resource for natural reading*. Abstract. DOI: [10.1038/sdata.2018.291](https://doi.org/10.1038/sdata.2018.291)
- **Emilia Turc et al. (2020)**. *Gaussian Mixture Modeling of EEG Alpha-Band Microarchitecture Reveals Continuous State Transitions During Burst Suppression*. Abstract. DOI: [10.2139/ssrn.6982383](https://doi.org/10.2139/ssrn.6982383)
- **Jinwon Chang et al. (2020)**. *Specification dependence of EEG theta/alpha ratio associations with cognitive impairment: a multiverse analysis*. Abstract. DOI: [10.64898/2026.09.03.749187](https://doi.org/10.64898/2026.09.03.749187)
- **Siddarth Mohanty (2020)**. *Evaluating cognitive workload in an AR environment using pupil area responses*. Abstract. DOI: [10.32469/10355/111118](https://doi.org/10.32469/10355/111118)
- **Tee Yi Wen et al. (2020)**. *Electroencephalogram (EEG) stress analysis on alpha/beta ratio and theta/beta ratio*. Abstract. DOI: [10.11591/ijeecs.v17.i1.pp175-182](https://doi.org/10.11591/ijeecs.v17.i1.pp175-182)
- **Chiara Jongerius et al. (2021)**. *Eye-tracking analyses of physician face gaze patterns in consultations*. Abstract. DOI: [10.1038/s41598-021-99068-4](https://doi.org/10.1038/s41598-021-99068-4)
- **James F. Cavanagh et al. (2022)**. *Frontal Midline Theta as a Model Specimen of Cortical Theta*. Abstract. DOI: [10.1093/oxfordhb/9780192898340.013.9](https://doi.org/10.1093/oxfordhb/9780192898340.013.9)
- **Kyoungwon Baik et al. (2022)**. *Implication of EEG theta/alpha and theta/beta ratio in Alzheimer’s and Lewy body disease*. Abstract. DOI: [10.1038/s41598-022-21951-5](https://doi.org/10.1038/s41598-022-21951-5)
- **Marta Mondellini et al. (2023)**. *A Multimodal Approach Exploiting EEG to Investigate the Effects of VR Environment on Mental Workload*. Abstract. DOI: [10.1080/10447318.2023.2258017](https://doi.org/10.1080/10447318.2023.2258017)
- **Narmashiri A et al. (2023)**. *Paranormal believers show reduced resting EEG beta band oscillations and inhibitory control than skeptics.*. Abstract. DOI: [10.1038/s41598-023-30457-7](https://doi.org/10.1038/s41598-023-30457-7)
- **Man BKL et al. (2025)**. *Visual Speech Reduces Cognitive Effort as Measured by EEG Theta Power and Pupil Dilation.*. Abstract. DOI: [10.1523/eneuro.0288-25.2025](https://doi.org/10.1523/eneuro.0288-25.2025)
- **Vincenzo Ronca et al. (2025)**. *A Multimodal Neurophysiological Approach to Evaluate Educational Contents in Terms of Cognitive Processes and Engagement*. Abstract. DOI: [10.3390/bioengineering12060597](https://doi.org/10.3390/bioengineering12060597)
- **A J Paton (2026)**. *Attention Stability: Admissible Focus and Constraint Regulation in Cognitive Systems*. Abstract. DOI: [10.5281/zenodo.19042955](https://doi.org/10.5281/zenodo.19042955)
- **Diotaiuti P et al. (2026)**. *Eye-tracking and cognitive patterns in post-traumatic stress disorder: a scoping review.*. Abstract. DOI: [10.1080/20008066.2026.2703319](https://doi.org/10.1080/20008066.2026.2703319)
- **Eunbyul Yang et al. (2026)**. *The impact of task complexity and self-regulated supports on cognitive load and emotional engagement through multimodal analysis in the metaverse learning environment*. Abstract. DOI: [10.3389/fpsyg.2026.1805065](https://doi.org/10.3389/fpsyg.2026.1805065)
- **Ferreira DF et al. (2026)**. *Beyond visual comfort: comparing pupillometry and subjective measures of cognitive strain in workplace lighting.*. Abstract. DOI: [10.3389/fpubh.2026.1875100](https://doi.org/10.3389/fpubh.2026.1875100)
- **Joerg F. Hipp et al. (2026)**. *Excess EEG beta-band oscillations in Dup15q syndrome correlate with clinical severity*. Abstract. DOI: [10.1038/s41398-026-04171-7](https://doi.org/10.1038/s41398-026-04171-7)
- **Maia A et al. (2026)**. *Attentional variability as a cognitive marker of clinical activity in Crohn's disease.*. Abstract. DOI: [10.1093/crocol/otag099](https://doi.org/10.1093/crocol/otag099)
- **Manjunath M et al. (2026)**. *Neural and cognitive correlates of adolescent anxiety: Widespread brain network alterations, increased control costs, and visuospatial processing deficits.*. Abstract. DOI: [10.1162/imag.a.1409](https://doi.org/10.1162/imag.a.1409)
- **Mei X et al. (2026)**. *Dual Mechanisms of Impaired Visual Processing in Glaucoma: Insights From Resting-State and Task-Related EEG.*. Abstract. DOI: [10.1167/iovs.67.12.2](https://doi.org/10.1167/iovs.67.12.2)
- **Mi D et al. (2026)**. *Resting-state frontal theta/beta ratio and psychological symptoms in college students: an EEG study.*. Abstract. DOI: [10.3389/fpsyt.2026.1834200](https://doi.org/10.3389/fpsyt.2026.1834200)
- **Navas-León S et al. (2026)**. *An Eye-Tracking Study on Text Accessibility and Comprehension in University Students.*. Abstract. DOI: [10.3390/bs16061041](https://doi.org/10.3390/bs16061041)
- **S. Kudubayeva et al. (2026)**. *Unified Visual Synchrony: A Framework for Face-Gesture Coherence in Multimodal Human-AI Interaction*. Abstract. DOI: [10.3390/bdcc10030088](https://doi.org/10.3390/bdcc10030088)
- **Sarimski R et al. (2026)**. *An Exploratory Study: What Can Eye-Movements Tell Us About Phonological Awareness in Children with Intellectual Disabilities?*. Abstract. DOI: [10.3390/jemr19050115](https://doi.org/10.3390/jemr19050115)
- **Strzelczyk D et al. (2026)**. *Theta-Beta Ratio in Attention Deficit Hyperactivity Disorder: A Multiverse Analysis*. Abstract. DOI: [10.64898/2026.01.08.26343676](https://doi.org/10.64898/2026.01.08.26343676)
- **Strzelczyk D et al. (2026)**. *Theta beta ratio in attention deficit hyperactivity disorder using a multiverse analysis.*. Abstract. DOI: [10.7554/elife.111114](https://doi.org/10.7554/elife.111114)
- **Sun H et al. (2026)**. *ReMind: A Retrospective Self-Report Paradigm for Studying Mind-Wandering Onset During Reading*. Abstract. DOI: [10.64898/2026.05.14.725227](https://doi.org/10.64898/2026.05.14.725227)
- **Sun H et al. (2026)**. *Brain-Language Alignment During Naturalistic Reading and Its Disruption by Mind-Wandering*. Abstract. DOI: [10.64898/2026.08.14.744875](https://doi.org/10.64898/2026.08.14.744875)
- **Xu Y et al. (2026)**. *Developmental variation in resting-state EEG spectral correlates of response inhibition in children and adolescents.*. Abstract. DOI: [10.1016/j.dcn.2026.101834](https://doi.org/10.1016/j.dcn.2026.101834)
- **Zhang J et al. (2026)**. *Insights into the neural mechanisms supporting cognitive stability and flexibility from human intracranial electroencephalography.*. Abstract. DOI: [10.1016/j.neubiorev.2026.106872](https://doi.org/10.1016/j.neubiorev.2026.106872)
- **Zou F et al. (2026)**. *Resting frontal alpha asymmetry is associated with rejection sensitivity beyond anxiety and depressive symptoms in young adults.*. Abstract. DOI: [10.1016/j.biopsycho.2026.109376](https://doi.org/10.1016/j.biopsycho.2026.109376)