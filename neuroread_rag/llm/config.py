"""
Configuration for the LLM personalized advisory module.
Supports Gemini API (cloud) and Ollama (local) backends.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Backend Selection: "gemini", "ollama", or "auto" (try ollama → gemini)
# --------------------------------------------------------------------------- #
LLM_BACKEND = os.environ.get("NEUROREAD_LLM_BACKEND", "auto")

# --------------------------------------------------------------------------- #
#  Gemini API Configuration (free tier)
# --------------------------------------------------------------------------- #
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.environ.get("NEUROREAD_GEMINI_MODEL", "gemini-2.5-flash")

# --------------------------------------------------------------------------- #
#  Ollama Configuration (local, offline)
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("NEUROREAD_OLLAMA_MODEL", "mistral")

# --------------------------------------------------------------------------- #
#  Generation Parameters
# --------------------------------------------------------------------------- #
LLM_TEMPERATURE = float(os.environ.get("NEUROREAD_LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("NEUROREAD_LLM_MAX_TOKENS", "4096"))

# --------------------------------------------------------------------------- #
#  Output cache directory (precomputed LLM reports persist here)
# --------------------------------------------------------------------------- #
LLM_CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "llm_reports"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Safety: banned clinical terms (post-processing filter)
# --------------------------------------------------------------------------- #
BANNED_CLINICAL_TERMS = [
    "adhd", "dyslexia", "disorder", "diagnosis", "diagnose", "clinical",
    "patient", "treatment", "medication", "pathology", "abnormal",
    "disease", "syndrome", "therapy", "psychiatric", "neurological disorder",
]
