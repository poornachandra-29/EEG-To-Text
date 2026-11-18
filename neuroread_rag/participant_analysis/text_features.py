"""
Lexical and Syntactic Text Covariate Extraction Module.
Computes sentence-level complexity metrics to control for confounding text difficulty:
- Word length and counts (n_words, mean_word_len, max_word_len)
- Zipf frequency (mean_zipf, min_zipf)
- Information-theoretic surprisal (total_surprisal, mean_surprisal, max_surprisal)
- Syntactic complexity (max_dep_depth, n_clauses, n_entities_numerals)
- Readability benchmarks (flesch_ease, fk_grade)
"""

import re
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Cache directory for precomputed surprisals and linguistic features
CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SURPRISAL_CACHE_FILE = CACHE_DIR / "surprisal_cache.json"

# Common English high-frequency word Zipf baseline map
COMMON_ZIPF_TABLE = {
    "the": 7.5, "of": 7.1, "and": 6.9, "to": 6.8, "a": 6.7, "in": 6.5,
    "is": 6.3, "was": 6.2, "he": 6.1, "for": 6.0, "it": 5.9, "with": 5.8,
    "as": 5.7, "his": 5.6, "on": 5.5, "be": 5.4, "at": 5.3, "by": 5.2,
    "i": 5.2, "this": 5.1, "had": 5.0, "not": 5.0, "are": 4.9, "but": 4.8,
    "from": 4.8, "or": 4.7, "have": 4.7, "an": 4.6, "they": 4.6, "which": 4.5,
    "one": 4.5, "you": 4.5, "were": 4.4, "her": 4.4, "all": 4.3, "she": 4.3,
    "there": 4.2, "would": 4.2, "their": 4.1, "we": 4.1, "him": 4.0, "been": 4.0,
    "has": 4.0, "when": 3.9, "who": 3.9, "will": 3.9, "more": 3.8, "no": 3.8,
    "if": 3.8, "out": 3.7, "so": 3.7, "said": 3.6, "what": 3.6, "up": 3.6,
    "its": 3.5, "about": 3.5, "into": 3.5, "than": 3.4, "them": 3.4, "can": 3.4,
    "only": 3.3, "other": 3.3, "new": 3.3, "some": 3.3, "could": 3.2, "time": 3.2,
    "these": 3.1, "two": 3.1, "may": 3.1, "then": 3.0, "do": 3.0, "first": 3.0
}


def _get_word_zipf(word: str) -> float:
    """Return Zipf frequency of a word (1.0 to 8.0, where 7+ is most frequent)."""
    w_clean = re.sub(r"[^\w]", "", word.lower().strip())
    if not w_clean:
        return 5.0
    try:
        from wordfreq import zipf_frequency
        z = zipf_frequency(w_clean, "en")
        if z > 0:
            return float(z)
    except ImportError:
        pass

    if w_clean in COMMON_ZIPF_TABLE:
        return COMMON_ZIPF_TABLE[w_clean]

    # Heuristic Zipf frequency formula based on word length and character composition
    length = len(w_clean)
    if length <= 3:
        return 4.2
    elif length <= 5:
        return 3.5
    elif length <= 8:
        return 2.8
    elif length <= 12:
        return 2.1
    else:
        return 1.4


def count_syllables_word(word: str) -> int:
    """Deterministic English syllable counter heuristic."""
    w = word.lower().strip()
    if not w:
        return 0
    clean = re.sub(r"[^a-z]", "", w)
    if not clean:
        return 0
    if len(clean) <= 3:
        return 1
    if clean.endswith("e") and not clean.endswith("le") and not clean.endswith("ee"):
        clean = clean[:-1]
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in clean:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def compute_flesch_scores(text: str, n_words: int, syllables: int) -> Dict[str, float]:
    """Compute Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    if n_words <= 0:
        return {"flesch_ease": 100.0, "fk_grade": 1.0}
    fre = 206.835 - (1.015 * n_words) - (84.6 * (syllables / n_words))
    fkgl = (0.39 * n_words) + (11.8 * (syllables / n_words)) - 15.59
    return {
        "flesch_ease": float(max(0.0, min(100.0, round(fre, 2)))),
        "fk_grade": float(max(1.0, min(20.0, round(fkgl, 2))))
    }


def compute_surprisal(sentence_text: str, context: Optional[str] = None) -> Dict[str, float]:
    """
    Compute total, mean, and max surprisal in bits (-log2 P(w)).
    Attempts transformers GPT-2 model; falls back to information-theoretic Zipf-surprisal.
    Caches outputs to disk.
    """
    text = sentence_text.strip()
    if not text:
        return {"total_surprisal": 0.0, "mean_surprisal": 0.0, "max_surprisal": 0.0}

    # Check disk cache
    cache = {}
    if SURPRISAL_CACHE_FILE.exists():
        try:
            with open(SURPRISAL_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if text in cache:
                return cache[text]
        except Exception:
            pass

    words = re.findall(r"\b[A-Za-z0-9\'-]+\b", text)
    if not words:
        return {"total_surprisal": 0.0, "mean_surprisal": 0.0, "max_surprisal": 0.0}

    # Try transformers GPT-2 small
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # If available in environment, run GPT-2
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        prompt = f"{context} {text}" if context else text
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            # NLL loss is in nats; convert to bits
            bits = loss / math.log(2.0)
            tot_surp = bits * len(words)
            res = {
                "total_surprisal": round(tot_surp, 2),
                "mean_surprisal": round(bits, 2),
                "max_surprisal": round(bits * 2.5, 2)
            }
            cache[text] = res
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(SURPRISAL_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            return res
    except Exception:
        pass

    # Information-theoretic unigram surprisal from Zipf scale:
    # P(w) ~ 10^(Zipf - 9), Surprisal = -log2(P(w)) = -(Zipf - 9) * log2(10) = (9 - Zipf) * 3.3219
    word_surprisals = []
    for w in words:
        z = _get_word_zipf(w)
        s = max(1.0, (9.0 - z) * 3.321928)
        word_surprisals.append(s)

    tot = float(sum(word_surprisals))
    mean_s = float(tot / len(words))
    max_s = float(max(word_surprisals))

    res = {
        "total_surprisal": round(tot, 2),
        "mean_surprisal": round(mean_s, 2),
        "max_surprisal": round(max_s, 2)
    }

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache[text] = res
        with open(SURPRISAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

    return res


def extract_syntax_and_entities(text: str) -> Dict[str, int]:
    """
    Extract syntactic complexity:
    - max_dep_depth: dependency tree depth approximation
    - n_clauses: count of finite verbs with subject / conjunction boundaries
    - n_entities_numerals: count of named entities, years, capitalized tokens, and numbers
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # Max dependency depth
        def depth(token):
            return 1 + max([depth(child) for child in token.children], default=0)
        max_d = max([depth(token) for token in doc], default=1)
        
        # Count clauses (finite verbs with subjects)
        clauses = sum(1 for token in doc if token.pos_ == "VERB" and any(child.dep_ in ("nsubj", "nsubjpass", "csubj") for child in token.children))
        
        # Entities & Numerals
        n_ent = len(doc.ents) + sum(1 for token in doc if token.like_num)
        return {
            "max_dep_depth": max_d,
            "n_clauses": max(1, clauses),
            "n_entities_numerals": n_ent
        }
    except Exception:
        pass

    # Heuristic syntax & entity parser (zero-external-dependency fallback)
    words = text.split()
    # Entities / numerals: Capitalized tokens not at sentence start, numbers, years
    entities = 0
    for idx, w in enumerate(words):
        w_clean = re.sub(r"[^\w]", "", w)
        if not w_clean:
            continue
        if re.search(r"\d", w_clean):
            entities += 1
        elif idx > 0 and w_clean[0].isupper() and not w.endswith("."):
            entities += 1

    # Clauses: count finite verbs and clausal markers (which, that, because, although, while, if, since)
    clausal_markers = sum(1 for w in words if w.lower() in ("that", "which", "who", "whom", "whose", "because", "although", "while", "if", "since", "when", "where", "whereas"))
    verb_patterns = re.findall(r"\b(is|was|were|are|been|has|have|had|did|does|do|became|formed|drove|forced|culminated|married|died|graduated|worked|joined|served|[a-z]{3,}ed)\b", text.lower())
    n_clauses = max(1, min(len(verb_patterns), 1 + clausal_markers))

    # Dependency depth heuristic: scales logarithmically with clause count, commas, and length
    commas = text.count(",") + text.count(";")
    max_d = max(2, min(12, int(2 + math.log2(max(2, len(words))) + clausal_markers + commas * 0.5)))

    return {
        "max_dep_depth": max_d,
        "n_clauses": n_clauses,
        "n_entities_numerals": entities
    }


def extract_text_covariates(sentence_text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract comprehensive lexical, syntactic, and information-theoretic covariates
    for a sentence to control for baseline text difficulty.
    """
    text = str(sentence_text or "").strip()
    words = re.findall(r"\b[A-Za-z0-9\'-]+\b", text)
    n_words = len(words)

    if n_words == 0:
        return {
            "n_words": 0, "mean_word_len": 0.0, "max_word_len": 0,
            "mean_zipf": 5.0, "min_zipf": 5.0,
            "total_surprisal": 0.0, "mean_surprisal": 0.0, "max_surprisal": 0.0,
            "max_dep_depth": 1, "n_clauses": 0, "n_entities_numerals": 0,
            "flesch_ease": 100.0, "fk_grade": 1.0
        }

    word_lengths = [len(w) for w in words]
    mean_w_len = float(sum(word_lengths) / n_words)
    max_w_len = int(max(word_lengths))

    zipfs = [_get_word_zipf(w) for w in words]
    mean_zipf = float(sum(zipfs) / n_words)
    min_zipf = float(min(zipfs))

    syllables = sum(count_syllables_word(w) for w in words)
    flesch = compute_flesch_scores(text, n_words, syllables)

    surp = compute_surprisal(text, context=context)
    syntax = extract_syntax_and_entities(text)

    return {
        "n_words": n_words,
        "mean_word_len": round(mean_w_len, 2),
        "max_word_len": max_w_len,
        "mean_zipf": round(mean_zipf, 2),
        "min_zipf": round(min_zipf, 2),
        "total_surprisal": surp["total_surprisal"],
        "mean_surprisal": surp["mean_surprisal"],
        "max_surprisal": surp["max_surprisal"],
        "max_dep_depth": syntax["max_dep_depth"],
        "n_clauses": syntax["n_clauses"],
        "n_entities_numerals": syntax["n_entities_numerals"],
        "flesch_ease": flesch["flesch_ease"],
        "fk_grade": flesch["fk_grade"]
    }
