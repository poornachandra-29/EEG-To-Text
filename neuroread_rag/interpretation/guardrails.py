"""
Scientific Guardrails and Safety Validation.
Prevents diagnostic overreach, unsubstantiated psychological claims, and invented universal ranges.
"""

import re
from typing import List, Tuple


# Patterns indicating affirmative clinical claims (forbidden) vs permitted negation disclaimers
AFFIRMATIVE_DIAGNOSTIC_PATTERNS = [
    r"\b(?:is|has been|are)\s+diagnos(?:ed|ing|ed)\b",
    r"\bdiagnos(?:ed|is|ing|e)\s+(?:of|with|as)\b",
    r"\b(?:confirmed|indicates?)\s+patholog(?:y|ical)\b",
    r"\bclinically\s+(?:impaired|deficient|abnormal|impairment)\b",
    r"\b(?:has|suffers from|manifests)\s+(?:adhd|add|autism)\b",
    r"\bdiagnosed\s+with\s+adhd\b",
    r"\bproves\s+mind[- ]wandering\b",
    r"\bdirectly\s+measures?\s+comprehension\b",
    r"\buniversal\s+normal\s+range\b",
    r"\bpathological\s+(?:brain|reading|eeg)\b"
]

PSYCHOLOGICAL_CONFLATION_REPLACEMENTS = [
    (r"\bproves mind[- ]wandering\b", "indicates reading cadence variability"),
    (r"\bclinical fatigue\b", "fatigue-compatible physiological pattern"),
    (r"\blacks comprehension\b", "encounters elevated processing demand"),
    (r"\babnormal reading\b", "reading pattern deviating from personal session baseline"),
]


class ScientificGuardrailValidator:
    """Enforces non-diagnostic compliance and scientific reporting standards."""

    @staticmethod
    def validate_text(text: str) -> Tuple[bool, List[str]]:
        """Verify that text is free of clinical diagnoses and diagnostic overreach."""
        violations = []
        text_lower = text.lower()
        for pat in AFFIRMATIVE_DIAGNOSTIC_PATTERNS:
            matches = re.findall(pat, text_lower)
            if matches:
                violations.append(f"Forbidden diagnostic/overreach term detected matching pattern: '{pat}'")

        is_valid = len(violations) == 0
        return is_valid, violations

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text to replace conflated psychological claims with physiological descriptors."""
        sanitized = text
        for pat, replacement in PSYCHOLOGICAL_CONFLATION_REPLACEMENTS:
            sanitized = re.sub(pat, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized
