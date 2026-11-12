"""
LLM Client Abstraction — dual-backend support for Gemini API and Ollama.

Automatically detects the available backend at startup:
  1. If Ollama is running locally → use it (fully offline, private)
  2. Else if GEMINI_API_KEY is set → use Gemini API (cloud, free tier)
  3. Else → raise an informative error with setup instructions
"""

import json
import logging
from typing import Optional, Dict, Any

from neuroread_rag.llm.config import (
    LLM_BACKEND,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    BANNED_CLINICAL_TERMS,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified interface to generate text from either Gemini API or Ollama."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or LLM_BACKEND
        self._resolved_backend: Optional[str] = None
        self._gemini_client = None
        self._resolve_backend()

    # ------------------------------------------------------------------ #
    #  Backend Detection
    # ------------------------------------------------------------------ #
    def _resolve_backend(self) -> None:
        if self.backend == "ollama" or self.backend == "auto":
            if self._check_ollama():
                self._resolved_backend = "ollama"
                logger.info(
                    "LLM Backend: Ollama (%s) at %s", OLLAMA_MODEL, OLLAMA_BASE_URL
                )
                return

        if self.backend == "gemini" or self.backend == "auto":
            if GEMINI_API_KEY:
                self._resolved_backend = "gemini"
                logger.info("LLM Backend: Gemini API (%s)", GEMINI_MODEL)
                return

        if self.backend == "auto":
            raise RuntimeError(
                "No LLM backend available.\n\n"
                "Option 1 — Ollama (offline):\n"
                "  curl -fsSL https://ollama.com/install.sh | sh\n"
                "  ollama pull mistral\n\n"
                "Option 2 — Gemini API (free tier):\n"
                "  1. Get a free key at https://aistudio.google.com/\n"
                "  2. export GEMINI_API_KEY='your-key-here'\n"
            )
        else:
            raise RuntimeError(
                f"Requested LLM backend '{self.backend}' is not available."
            )

    def _check_ollama(self) -> bool:
        """Check if Ollama server is running and the model is available."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                model_names = [
                    m.get("name", "").split(":")[0]
                    for m in data.get("models", [])
                ]
                if OLLAMA_MODEL in model_names or any(
                    OLLAMA_MODEL in n for n in model_names
                ):
                    return True
                logger.warning(
                    "Ollama running but model '%s' not found. Available: %s",
                    OLLAMA_MODEL,
                    model_names,
                )
                return False
        except Exception:
            return False

    @property
    def active_backend(self) -> str:
        return self._resolved_backend or "none"

    @property
    def model_name(self) -> str:
        if self._resolved_backend == "gemini":
            return GEMINI_MODEL
        elif self._resolved_backend == "ollama":
            return OLLAMA_MODEL
        return "none"

    # ------------------------------------------------------------------ #
    #  Main Generation Interface
    # ------------------------------------------------------------------ #
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the active LLM backend."""
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

        if self._resolved_backend == "gemini":
            raw = self._gemini_generate(system_prompt, user_prompt, temp, max_tok)
        elif self._resolved_backend == "ollama":
            raw = self._ollama_generate(system_prompt, user_prompt, temp, max_tok)
        else:
            raise RuntimeError("No LLM backend resolved.")

        return self._safety_filter(raw)

    # ------------------------------------------------------------------ #
    #  Gemini API Backend
    # ------------------------------------------------------------------ #
    def _gemini_generate(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        try:
            from google import genai
            from google.genai import types

            if self._gemini_client is None:
                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)

            response = self._gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
        except ImportError:
            raise RuntimeError(
                "Gemini SDK not installed. Run: pip install google-genai"
            )
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise

    # ------------------------------------------------------------------ #
    #  Ollama Backend
    # ------------------------------------------------------------------ #
    def _ollama_generate(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        import urllib.request

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "")
        except Exception as e:
            logger.error("Ollama error: %s", e)
            raise

    # ------------------------------------------------------------------ #
    #  Post-Processing Safety Filter
    # ------------------------------------------------------------------ #
    def _safety_filter(self, text: str) -> str:
        """Remove or flag clinical diagnostic language that the LLM should not produce."""
        lower_text = text.lower()
        flagged = [term for term in BANNED_CLINICAL_TERMS if term in lower_text]
        if flagged:
            logger.warning(
                "LLM output contained banned clinical terms: %s — adding disclaimer.",
                flagged,
            )
            text += (
                "\n\n---\n"
                "> **⚠️ Disclaimer**: This report is an educational learning "
                "optimization tool based on reading-task physiological proxies. "
                "It does NOT constitute medical, psychiatric, or clinical assessment. "
                "Consult qualified professionals for any health-related concerns."
            )
        return text
