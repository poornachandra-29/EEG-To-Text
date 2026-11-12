"""
LLM-Enhanced Personalized Learning Advisory Module.

Integrates Large Language Models (Gemini API / Ollama) with the NeuroRead RAG
pipeline to generate personalized, evidence-based study recommendations from
each student's EEG + eye-tracking cognitive profile.
"""

from neuroread_rag.llm.personalized_advisor import PersonalizedAdvisor

__all__ = ["PersonalizedAdvisor"]
