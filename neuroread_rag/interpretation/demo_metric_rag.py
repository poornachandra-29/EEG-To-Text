"""
Interactive CLI Demonstrator for Pure RAG LLM Metric-Specific Assessment.
Demonstrates end-to-end Pure RAG execution:
1. Retrieval of peer-reviewed papers from vector database (scientific_literature.db)
2. Empirical review of student CSV recording (baseline, deviations, friction trials)
3. LLM prompt construction
4. Generation of the 4 personalized pedagogical sections
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from neuroread_rag.participant_analysis.analyzer import ParticipantAnalyzer
from neuroread_rag.ingestion.vector_store import HybridVectorStore
from neuroread_rag.retrieval.synthesizer import EvidenceSynthesizer
from neuroread_rag.interpretation.llm_metric_rag import LLMMetricRAGGenerator


def run_demo(csv_path: str, target_metric: str = "EEI", show_prompt: bool = True):
    print("=" * 80)
    print("       NEUROREAD-RAG: PURE RAG LLM METRIC ASSESSMENT DEMONSTRATOR")
    print("=" * 80)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"[Error] CSV file not found: {csv_file}")
        sys.exit(1)

    print(f"\n📂 Step 1: Loading Student CSV Recording: {csv_file.name}")
    analyzer = ParticipantAnalyzer(csv_file)
    participant_result = analyzer.analyze()
    print(f"   Participant ID: {participant_result.participant_id}")
    print(f"   Total Sentences Analyzed: {participant_result.total_sentences}")

    if target_metric not in participant_result.metrics:
        available = list(participant_result.metrics.keys())
        print(f"[Error] Metric '{target_metric}' not found in CSV. Available metrics: {available}")
        sys.exit(1)

    summary = participant_result.metrics[target_metric]
    print(f"\n📊 Step 2: Student Empirical Measurements for {target_metric}:")
    print(f"   - Session Mean: {summary.mean:.3f} {summary.unit}")
    print(f"   - Session Baseline: {summary.session_baseline:.3f} {summary.unit}")
    print(f"   - Relative Deviation: {summary.relative_deviation_pct:+.1f}% ({summary.deviation_category})")
    print(f"   - Variability Index: {summary.variability_index:.3f}")

    print(f"\n🔬 Step 3: Retrieving Scientific Literature from Vector DB (scientific_literature.db)...")
    vs = HybridVectorStore()
    synthesizer = EvidenceSynthesizer(vs)
    evidence = synthesizer.synthesize_metric(target_metric)
    print(f"   Consensus Level: {evidence.scientific_evidence_level}")
    print(f"   Retrieved {len(evidence.papers_cited)} Cited Papers:")
    for i, p in enumerate(evidence.papers_cited, 1):
        doi_str = f" [DOI: {p.doi}]" if p.doi else ""
        print(f"   [{i}] {p.authors} ({p.year}) — \"{p.paper_title}\"{doi_str}")

    print(f"\n🤖 Step 4: Invoking Pure RAG LLM Generation Engine...")
    rag_gen = LLMMetricRAGGenerator(vector_store=vs, synthesizer=synthesizer)
    df = getattr(analyzer, "df", None)
    if df is None:
        df = pd.read_csv(csv_file)

    spec = rag_gen.generate(
        metric_id=target_metric,
        metric_summary=summary,
        sentence_observations=participant_result.sentence_observations,
        df=df,
        synthesized_evidence=evidence,
    )
    print(f"   Active Provider: {spec['llm_provider']}")

    if show_prompt:
        print("\n" + "-" * 80)
        print("📝 EXACT PROMPT SENT TO LLM:")
        print("-" * 80)
        print(spec["llm_prompt_used"].strip())
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"✨ LLM-GENERATED 4-POINT PERSONALIZED PEDAGOGICAL INSTRUCTIONS FOR: {target_metric}")
    print("=" * 80)

    print("\n🎯 POINT 1: WHAT THIS METRIC SAYS (SCIENTIFIC GROUND TRUTH):")
    print(spec["what_this_metric_says"])

    print("\n👤 POINT 2: HOW YOU ARE PERFORMING ACCORDING TO THIS METRIC (CSV REVIEW):")
    print(spec["how_you_are_performing"])

    diag = spec["where_did_you_make_a_mistake"]
    print(f"\n⚠️ DIAGNOSED FRICTION TRIAL (Sentence #{diag.get('sentence_index')}):")
    print(f"   Sentence: \"{diag.get('sentence_text')}\"")
    print(f"   Linguistics: Grade Level {diag.get('grade_level', 0):.1f} | Flesch Ease: {diag.get('flesch_reading_ease', 0):.1f} | Complex Words: {diag.get('complex_words_pct', 0):.0f}%")
    print(f"   Observed Metric: {diag.get('metric_value')} (Robust Z: {diag.get('robust_z', 0):+.2f})")
    print(f"   Cognitive Root Cause: {diag.get('diagnosis_note')}")

    print("\n💡 POINT 3: HOW TO IMPROVE RESULTS IN THE CONTEXT OF THIS METRIC (ACTIONABLE STRATEGY):")
    for r in spec["how_to_improve"]:
        print(f"   - {r}")

    print("\n🚀 POINT 4: WHAT IMPROVING THIS METRIC ACTUALLY MEANS TO YOU (COGNITIVE IMPACT):")
    print(spec["what_improving_means_to_you"])

    print("\n" + "=" * 80)
    print("✅ PURE RAG DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure RAG LLM Metric Assessment Demonstrator")
    parser.add_argument(
        "--csv",
        type=str,
        default="neuroread_rag/data/precomputed_csvs/task2_NR/features_task2_NR_ZPH.csv",
        help="Path to participant CSV file",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="EEI",
        help="Target metric to analyze (e.g. EEI, FMT, TRT, GD, PupilSize, etc.)",
    )
    parser.add_argument(
        "--hide-prompt",
        action="store_true",
        help="Hide the prompt sent to the LLM",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional free API key (Groq, Gemini, or OpenRouter) to override config",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["groq", "gemini", "openrouter"],
        help="LLM provider (default: auto-detected from key or groq)",
    )
    args = parser.parse_args()

    if args.api_key:
        import neuroread_rag.interpretation.llm_metric_rag as lmr
        lmr.FREE_LLM_API_KEY = args.api_key
        if args.provider:
            lmr.FREE_LLM_PROVIDER = args.provider

    run_demo(csv_path=args.csv, target_metric=args.metric, show_prompt=not args.hide_prompt)
