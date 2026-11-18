"""
End-to-End Reproducible Pipeline Coordinator for NeuroRead RAG.
Runs all pipeline stages deterministically with random seed 42.
"""

import sys
import time
from pathlib import Path

from neuroread_rag.utils.seeds import set_seed
from neuroread_rag.participant_analysis.residualize import ConfoundResidualizer, plot_residualization_diagnostic
from neuroread_rag.participant_analysis.sentence_comparator import SentenceComparator
from neuroread_rag.validation.predictive_models import ModelValidator
from neuroread_rag.cluster_machine_learning.cluster_rework import ClusterReworkPipeline


def run_pipeline() -> None:
    start_time = time.time()
    print("=" * 70)
    print("🧠 Starting NeuroRead RAG Reproducible Pipeline (Seed: 42)")
    print("=" * 70)

    # 1. Deterministic Multi-Library Seeding
    set_seed(42)
    print("\n[Step 1/5] Global multi-library seeds set to 42.")

    # 2. Confound Residualization on Cohort Data
    print("\n[Step 2/5] Running lexical and syntactic confound residualization...")
    import glob
    import pandas as pd
    cohort_files = sorted(glob.glob("neuroread_rag/data/precomputed_csvs/task2_NR/features_task2_NR_*.csv"))
    if not cohort_files:
        print("  Error: No participant CSV files found in neuroread_rag/data/precomputed_csvs/task2_NR/")
        sys.exit(1)

    dfs = [pd.read_csv(f) for f in cohort_files]
    cohort_df = pd.concat(dfs, ignore_index=True)
    if "theta_fm" not in cohort_df.columns and "FMT" in cohort_df.columns:
        cohort_df["theta_fm"] = cohort_df["FMT"]

    res = ConfoundResidualizer(random_state=42)
    cohort_loso = res.cross_fit_loso(cohort_df, subject_col="patient_id")
    res.fit(cohort_df)
    res.save()
    print(f"  Processed {len(cohort_loso)} sentences across {len(cohort_files)} participants.")

    # 3. Reading State & Behavioral Breakdown Labeling
    print("\n[Step 3/5] Assigning behavioral breakdown and external comprehension labels...")
    comp = SentenceComparator(cohort_loso)
    cohort_labeled = comp.classify_reading_states(cohort_loso)

    stages_dir = Path("neuroread_rag/artifacts/stages")
    stages_dir.mkdir(parents=True, exist_ok=True)
    cohort_labeled_path = stages_dir / "stage3_cohort_processed.csv"
    cohort_labeled.to_csv(cohort_labeled_path, index=False)
    print(f"  Saved labeled cohort dataset to: {cohort_labeled_path}")

    # Generate residualization diagnostic figure
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_residualization_diagnostic(cohort_labeled, figures_dir / "trt_vs_surprisal_residualization.png")
    print("  Saved figures/trt_vs_surprisal_residualization.png")

    # 4. Predictive Modeling & Metric Pruning (Phase 4)
    print("\n[Step 4/5] Executing cross-modal predictive validation and metric pruning...")
    validator = ModelValidator(cohort_labeled_path)
    val_results = validator.run_validation()
    print("  Saved reports/metric_importance.csv and reports/validation_report.md")

    # 5. Clustering Architecture & BIC Model Selection
    print("\n[Step 5/5] Running GMM clustering and BIC model selection...")
    cluster_pipe = ClusterReworkPipeline(cohort_labeled_path)
    cluster_results = cluster_pipe.run_sweep_and_fit()
    print(f"  Selected K={cluster_results['selected_k']} ({cluster_results['selected_covariance']} covariance).")
    print(f"  Model BIC: {cluster_results['model_bic']:.1f} | AIC: {cluster_results['model_aic']:.1f}")
    print("  Saved reports/cluster_selection.csv and neuroread_rag/artifacts/cluster_profiles.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ NeuroRead RAG Pipeline execution completed successfully in {elapsed:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
