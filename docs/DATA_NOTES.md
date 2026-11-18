# 📖 ZuCo 1.0 Dataset Inspection & Engineering Notes
**Reference Documentation for NeuroRead RAG Pipeline**

---

## 1. Corpus Overview
- **Source**: ZuCo 1.0 (Hollenstein et al., 2018, *Scientific Data*).
- **Participants**: 12 healthy adult native English speakers (5 female, 7 male, ages 22–54, all right-handed).
  - Subject IDs: `ZAB`, `ZDM`, `ZDN`, `ZGW`, `ZJM`, `ZJN`, `ZJS`, `ZKB`, `ZKH`, `ZKW`, `ZMG`, `ZPH`.
- **Tasks**:
  1. `task1_SR` (Sentiment Reading): 400 sentences from movie reviews with manual positive/negative/neutral labels.
  2. `task2_NR` (Natural Reading): 300 sentences from Wikipedia biographical articles read naturally.
  3. `task3_TSR` (Task-Specific Reading): 407 sentences where participants search for specific relation types.

---

## 2. Hardware, Electrodes & Montage Specifications
- **EEG System**: 128-channel EGI HydroCel Geodesic Sensor Net sampled at 500 Hz with online reference Cz (electrode E129).
- **Preprocessing in ZuCo**: Automated artifact removal via Automagic toolbox.
- **Channel Selection**:
  - Out of 128 electrodes, 23 outer/neck/face channels were removed by the original authors due to muscle artifacts, leaving **105 scalp channels**.
  - In addition, 9 dedicated EOG electrodes were placed for ocular movement monitoring.
  - The sentence-level `rawData` matrix distributed in ZuCo has dimensions `(105, n_samples)`.
- **Frontal Midline Theta (FMT) Channel Set**:
  - In the EGI 128-channel net, the canonical 10-10 frontal-midline positions correspond to:
    - **Fz** $\approx$ **E11** (Index `10` in 0-indexed 105 array).
    - **FCz** $\approx$ **E6** (Index `5` in 0-indexed 105 array).
  - In `neuroread_rag v2.0`, `theta_fm` is computed specifically across channels `[E6, E11]` (indices `[5, 10]`), while `theta_global` retains the whole-scalp average across all 105 channels.

---

## 3. Sentence Duration & Reliability Thresholds
- **Sampling Rate**: 500 Hz (1 sample = 2 ms).
- **Sentence Duration Distribution**:
  - Sentences range from under 1.5 seconds to over 10 seconds.
  - For Welch PSD estimation:
    - **Full Reliability ($\ge 4.0$ s / 2000 samples)**: Computed with standard `nperseg = 512`. Flagged as `eeg_reliability = "full"`.
    - **Low Reliability ($2.0 \le t < 4.0$ s)**: Computed with smaller segment `nperseg = 256`. Flagged as `eeg_reliability = "low"`.
    - **Insufficient Signal ($t < 2.0$ s)**: Spectral estimates set to `NaN`. Flagged as `eeg_reliability = "insufficient"`. Downstream models skip or down-weight these rows and never impute with zeros.

---

## 4. Control & Comprehension Questions
- **Task 2 (Natural Reading)**:
  - Contains **36 control questions** presented pseudo-randomly across the 300 sentences (about 12% of trials) to verify comprehension.
  - Questions and options are recorded in `relations_normal_reading_control_questions.csv`.
  - Participant responses are recorded in `answers/Fullresults_NR_<pid>.mat` as an $N \times 3$ array:
    - Column 0: `paragraph_id`
    - Column 1: `sentence_id`
    - Column 2: chosen option (1, 2, or 3).
  - External outcome label `comprehension_error`:
    - `0`: Correct answer to control question.
    - `1`: Incorrect answer.
    - `NaN`: Unqueried sentences (the remaining ~88% of trials).
- **Task 3 (Task-Specific Reading)**:
  - Participants provide binary judgements (relation present vs absent) for all 407 sentences.

---

## 5. Missing Data & Incomplete Blocks
- In Task 2 (NR), sentences 50–99 are absent from the released data for participants **ZPH** and **ZJS**.
- In participant ZPH, three further sentences (203, 214, 264) contain eye-tracking records but missing EEG segments.
- After removing rows with all features missing, Task 2 (NR) contains 3,493 usable sentences (3,243 across 11 training participants, 250 for held-out ZPH).
