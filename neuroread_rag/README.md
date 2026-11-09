# NeuroRead AI: Research-Grounded RAG Pipeline

A research-grounded Retrieval-Augmented Generation (RAG) module that integrates with the ZuCo simultaneous EEG and eye-tracking natural reading pipeline.

---

## 🌟 Key Architecture & Highlights

* **100% Non-Invasive Integration**: Your existing ML pipeline (`train.py`, `test.py`, `cognitive_profile_2.py`) is completely untouched.
* **Deterministic Numerical Processing**: Programmatic pandas/numpy statistics (means, medians, standard deviations, MAD, robust SD, relative baseline deviations, and sentence outlier tracking) with **zero LLM calculation or hallucination**.
* **Multi-Source Research Discovery**: Aggregates peer-reviewed literature across **OpenAlex, Semantic Scholar, Europe PMC, PubMed, and Crossref**, cross-referencing canonical literature (*ZuCo Nature Scientific Data, Pope et al., Cavanagh & Frank, Rayner, Beatty & Kahneman, Dimigen et al., Klimesch*).
* **Section-Aware Hybrid Vector RAG**: Combines BM25 keyword matching with dense semantic cosine similarity (Reciprocal Rank Fusion) and metadata filtering.
* **Contradiction & Nuance Detection**: Flags context-dependent divergence (e.g. FMT cognitive control vs. fatigue, pupil dilation mental effort vs. constriction/drift).
* **Personalized Recommendations**: Concrete reading and cognitive pacing strategies backed by real citations and verified DOIs.
* **Standardized 10-Section Assessment Report**: Available in Markdown (`.md`), JSON (`.json`), and self-contained printable HTML (`.html`).
* **Interactive Web Dashboard**: Modern UI in pure HTML, CSS, and Vanilla JavaScript with interactive charts, live Research Explorer, and PDF printing.

---

## 📂 Directory Structure

```
EEG-To-Text/
└── neuroread_rag/
    ├── config.py                     # Metric definitions, units, search keywords, guardrails
    ├── main.py                       # CLI entrypoint for report generation and server launch
    ├── server.py                     # Lightweight REST API server (Python standard http.server)
    ├── participant_analysis/         # Phase 1: Pure deterministic pandas/numpy analytics
    │   ├── analyzer.py               # Deterministic stats, baselines, sentence outliers
    │   └── data_models.py            # Typed dataclasses for metrics, clusters, observations
    ├── discovery/                    # Phase 2: Scientific paper discovery
    │   ├── models.py                 # PaperMetadata, ResearchChunk schemas
    │   ├── openalex.py               # OpenAlex API client
    │   ├── semantic_scholar.py       # Semantic Scholar Graph API client
    │   ├── pubmed.py                 # PubMed E-Utilities & Europe PMC clients
    │   ├── crossref.py               # Crossref DOI verification client
    │   └── aggregator.py             # Multi-source paper aggregator and deduplicator
    ├── ingestion/                    # Phase 2: Ingestion & Vector Indexing
    │   ├── chunker.py                # Scientific section-aware chunker
    │   ├── vector_store.py           # Hybrid BM25 + dense semantic store (SQLite)
    │   ├── evidence_extractor.py     # Metric claim and directionality parser
    │   └── pipeline.py               # End-to-end ingestion pipeline coordinator
    ├── retrieval/                    # Phase 3: Hybrid Retrieval & Synthesis
    │   ├── hybrid_search.py          # Query expansion & hybrid retrieval
    │   ├── reranker.py               # Academic multi-factor reranker (reading context boost)
    │   ├── contradiction.py          # Contradictory & context-dependent nuance detector
    │   └── synthesizer.py            # Multi-paper consensus synthesizer
    ├── interpretation/               # Phase 4: Research-Grounded Interpretation
    │   ├── models.py                 # MetricInterpretation & GroundedReport schemas
    │   ├── guardrails.py             # Safety validators (enforces non-diagnostic compliance)
    │   └── grounding_engine.py       # Connects numerical measurements to literature
    ├── recommendations/              # Phase 5: Personalized Recommendations
    │   └── recommender.py            # Actionable guidance with citations & DOIs
    ├── reporting/                    # Phase 5: 10-Section Report Generator
    │   └── report_generator.py       # Markdown, HTML, and JSON report generator
    ├── frontend/                     # Phase 6: HTML, CSS, JavaScript Dashboard
    │   ├── index.html                # Responsive neurotech dashboard
    │   ├── style.css                 # Modern CSS with print media queries for PDF
    │   └── app.js                    # Vanilla JS client (charts, search, live filters)
    ├── output/                       # Generated artifacts, databases, reports
    │   ├── scientific_literature.db  # SQLite persistent vector database
    │   ├── extracted_evidence.json   # Extracted empirical claims catalog
    │   ├── resultsZPH_NR_analysis.json
    │   ├── resultsZPH_NR_grounded_report.json
    │   ├── resultsZPH_NR_final_report.json
    │   ├── resultsZPH_NR_final_report.md
    │   └── resultsZPH_NR_final_report.html
    └── tests/                        # Full 26-test unit and integration suite
        ├── test_participant_analysis.py
        ├── test_discovery_and_ingestion.py
        ├── test_retrieval.py
        ├── test_interpretation.py
        ├── test_recommendations_and_report.py
        └── test_server_and_frontend.py
```

---

## 🚀 Quickstart Guide

### 1. Generate Reports
To execute the pipeline and generate the 10-section report in Markdown, HTML, and JSON:
```bash
python -m neuroread_rag.main --generate-report
```

### 2. Launch the Interactive Dashboard Server
To start the local web dashboard and REST API server:
```bash
python -m neuroread_rag.main --server --port 8080
```
Open **`http://127.0.0.1:8080/`** in your browser to view:
* **Executive Overview**: Participant stats and summary cards.
* **Focus & Attention Report**: Interactive SVG distribution of cognitive patterns.
* **Cognitive Workload Report**: CWI multi-metric breakdown and cluster profiles.
* **Metric Scientific Assessment**: Filterable cards across Spectral EEG, Eye-Tracking, and Composite Proxies.
* **Research Evidence Comparison**: Live side-by-side comparison table.
* **Research Explorer**: Live search bar querying the vector store.
* **Personalized Recommendations**: Expandable science-backed pacing cards.
* **Complete AI Assessment Report**: Printable view with one-click **Print / Save as PDF**.

### 3. Run the Test Suite
To run all 26 unit and integration tests:
```bash
python -m unittest discover -s neuroread_rag/tests -v
```

---

## 🔒 Scientific Guardrails & Compliance

* **Physiological Observations vs. Psychological Interpretations**: Distinguishes objective signal modulations from subjective states.
* **Non-Diagnostic Standards**: Explicitly forbids diagnostic claims (e.g. ADHD, dyslexia, cognitive disorders).
* **No Universal Population Cutoffs**: Evaluates participants strictly against within-session personal baseline reference intervals.
* **Cadence Proxies**: Treats MWLI and SAI strictly as reading cadence irregularity proxies rather than proof of mind-wandering.
