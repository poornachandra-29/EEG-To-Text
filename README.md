# 🧠 NeuroRead: Interpretable Reading Engagement Metrics from EEG and Eye-Tracking Data

> Multi-modal Cognitive Reading Analytics, Physiological RAG Grounding, and EEG-to-Text Decoding System for the Zurich Cognitive Language Processing Corpus (ZuCo).

---

## 🚀 Quickstart: How to Run the Project

Follow these steps to navigate to the project directory, install dependencies, and run the pipeline or dashboard.

### Step 1: Navigate to the Repository Directory
Clone the repository (if not already cloned) and navigate into the project root directory:

```bash
# Clone the repository
git clone https://github.com/poornachandra-29/EEG-To-Text.git

# Go to the project root directory
cd EEG-To-Text
```

---

### Step 2: Install Requirements

Choose either the lightweight Python environment (for NeuroRead RAG & Web Dashboard) or the Conda environment (for full EEG decoding models):

#### Option A: Lightweight Python Dependencies (NeuroRead RAG, Server & Analytics)
```bash
pip install -r requirements.txt
```
*(Or specify the subfolder requirements: `pip install -r neuroread_rag/requirements.txt`)*



---

### Step 3: Run Commands

Depending on what component you wish to run:

#### 1. Launch Interactive Web Dashboard & REST API
```bash
python -m neuroread_rag.server --port 8080
```
Then open your web browser and navigate to:
```
http://localhost:8080
```

#### 2. Run Full NeuroRead Analysis & Generate Reports
```bash
python -m neuroread_rag.main --generate-report
```

#### 3. Run Cognitive Profile Analysis
```bash
python cognitive_profile.py
```

#### 4. Run Clustering & Machine Learning Suite
```bash
python neuroread_rag/cluster_machine_learning/test.py
```

#### 5. Run All Unit Tests
```bash
python -m unittest discover neuroread_rag/tests/
```

---

## 📂 Repository Layout

- `neuroread_rag/`: Multi-modal cognitive analytics, LLM metric grounding, interactive web dashboard, and clustering engine.
- `cognitive_profile.py`: Individual cognitive workload profiling, reference ranges, and state timeline visualizations.
- `scripts/`: Shell scripts for training and evaluating EEG-To-Text decoding and sentiment pipelines.
- `docs/`: Technical notes, model cards, and dataset documentation.

---

## Create Environment
run `conda env create -f environment.yml` to create the conda environment (named "EEGToText") used in our experiments.

## Download ZuCo datasets
- Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from https://osf.io/q3zws/files/ under 'OSF Storage' root,  
unzip and move all `.mat` files to `~/datasets/ZuCo/task1-SR/Matlab_files`,`~/datasets/ZuCo/task2-NR/Matlab_files`,`~/datasets/ZuCo/task3-TSR/Matlab_files` respectively.


## Preprocess datasets
run `bash ./scripts/prepare_dataset.sh` to preprocess `.mat` files and prepare sentiment labels. 

For each task, all `.mat` files will be converted into one `.pickle` file stored in `~/datasets/ZuCo/<task_name>/<task_name>-dataset.pickle`. 

Sentiment dataset for ZuCo (`sentiment_labels.json`) will be stored in `~/datasets/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json`. 

Sentiment dataset for filtered Stanford Sentiment Treebank will be stored in `~/datasets/stanfordsentiment/ternary_dataset.json`


## EEG-To-Text Decoding
To train an EEG-To-Text decoding model, run `bash ./scripts/train_decoding.sh`.

To evaluate the trained EEG-To-Text decoding model from above, run `bash ./scripts/eval_decoding.sh`.

For detailed configuration of the available arguments, please refer to function `get_config(case = 'train_decoding')` in `/config.py`

### Zero-shot sentiment classification pipeline 
We first train the decoder and the classifier individually, and then we evaluate the pipeline on ZuCo task1-SR data.

To run the whole training and evaluation process, run `bash ./scripts/train_eval_zeroshot_pipeline.sh`.

For detailed configuration of the available arguments, please refer to function `get_config(case = 'eval_sentiment')` in `/config.py`
