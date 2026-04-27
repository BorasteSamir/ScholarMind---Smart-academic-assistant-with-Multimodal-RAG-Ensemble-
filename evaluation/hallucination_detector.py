"""
evaluation/hallucination_detector.py
=====================================
Heuristic hallucination detection for RAG-generated answers.

Does NOT use another LLM call — uses fast, deterministic heuristics:

  1. Unsupported Named Entities:
     Checks if proper nouns in the answer exist in the retrieved context.
     Unverified entities are flagged as potential hallucinations.

  2. Numeric Consistency:
     Checks whether numbers in the answer appear in the context.

  3. Claim Coverage:
     Measures the lexical overlap between the answer and context.
     Low overlap = the answer may be "making things up."

  4. Hedge Word Check:
     Detects overconfident phrasing in low-confidence contexts.

The detector returns a (bool_flag, warning_string) tuple.
A True flag means hallucination is suspected.
"""

import logging
import re
from typing import Optional

from config.settings import HALLUCINATION_THRESHOLD

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HEURISTIC FUNCTIONS
# ─────────────────────────────────────────────

def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric literals from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _extract_proper_nouns(text: str) -> set[str]:
    """
    Simple proper noun extractor — capitalized words not at sentence start.
    Not perfect, but fast and dependency-free.
    """
    words = re.findall(r"(?<!\. )[A-Z][a-z]{2,}", text)
    # Exclude common stop-words and titles
    exclude = {
        "The", "This", "These", "That", "Those", "Here", "There",
        "In", "On", "At", "By", "For", "Of", "To", "And", "Or",
        "With", "From", "However", "Therefore", "Thus", "Hence",
        "Note", "Table", "Figure", "Section", "Chapter", "Algorithm",
    }
    return {w for w in words if w not in exclude}


def _token_overlap_ratio(answer: str, context: str) -> float:
    """
    Compute token-level overlap ratio:
      |tokens(answer) ∩ tokens(context)| / |tokens(answer)|

    A ratio below 0.3 suggests the answer is not grounded in context.
    """
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))

    # Remove very common English stop words
    stops = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "on", "at", "to", "for", "and", "or", "but",
        "not", "this", "that", "it", "its", "with", "from", "by",
        "as", "if", "than", "then", "so", "yet", "both", "each",
    }
    answer_tokens -= stops
    if not answer_tokens:
        return 1.0

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def _has_overconfident_phrasing(answer: str) -> bool:
    """
    Detect overconfident phrasing not supported by evidence.
    Phrases like "definitely", "certainly", "absolutely" without hedging.
    """
    overconfident_patterns = [
        r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
        r"\bwithout (a )?doubt\b", r"\bproven fact\b",
        r"\bit is obvious\b", r"\bclearly shows\b",
        r"\bthe answer is always\b", r"\bnever (fails|misses)\b",
    ]
    text_lower = answer.lower()
    return any(re.search(p, text_lower) for p in overconfident_patterns)


# ─────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────

class HallucinationDetector:
    """
    Heuristic hallucination detector for RAG-generated answers.

    Usage:
        detector = HallucinationDetector()
        flag, warning = detector.check(context="...", answer="...")
        if flag:
            print(f"⚠️ Potential hallucination: {warning}")
    """

    def __init__(self, threshold: float = HALLUCINATION_THRESHOLD):
        self.threshold = threshold

    def check(
        self,
        context: str,
        answer: str,
    ) -> tuple[bool, str]:
        """
        Run all heuristic checks and aggregate results.

        Args:
            context: The retrieved context text used for generation
            answer:  The LLM-generated answer

        Returns:
            (flag: bool, warning: str)
            flag = True if hallucination is suspected
            warning = human-readable explanation
        """
        if not context.strip() or not answer.strip():
            return False, ""

        issues: list[str] = []

        # ── Check 1: Token Overlap ──
        overlap = _token_overlap_ratio(answer, context)
        if overlap < self.threshold:
            issues.append(
                f"Low lexical overlap between answer and context ({overlap:.0%}). "
                "The answer may contain information not found in the document."
            )

        # ── Check 2: Numeric Consistency ──
        answer_nums = _extract_numbers(answer)
        context_nums = _extract_numbers(context)
        unsupported_nums = answer_nums - context_nums
        if unsupported_nums and len(unsupported_nums) > 2:
            issues.append(
                f"Numeric values not found in context: {', '.join(sorted(unsupported_nums)[:5])}. "
                "Verify these figures against the source document."
            )

        # ── Check 3: Unsupported Proper Nouns ──
        answer_nouns = _extract_proper_nouns(answer)
        context_nouns = _extract_proper_nouns(context)
        unsupported_nouns = answer_nouns - context_nouns
        if len(unsupported_nouns) > 3:
            sample = ", ".join(sorted(unsupported_nouns)[:5])
            issues.append(
                f"Proper nouns not found in context: {sample}. "
                "These may be hallucinated entities."
            )

        # ── Check 4: Overconfident Phrasing ──
        if _has_overconfident_phrasing(answer) and overlap < 0.5:
            issues.append(
                "Answer uses overconfident language despite low context overlap. "
                "Treat claims with caution."
            )

        # ── Aggregate ──
        if issues:
            warning = " | ".join(issues)
            logger.warning(f"Hallucination detected: {warning[:200]}")
            return True, warning

        return False, ""

    def score(self, context: str, answer: str) -> float:
        """
        Return a hallucination risk score (0 = safe, 1 = high risk).
        Useful for logging and dashboards.
        """
        overlap = _token_overlap_ratio(answer, context)
        # Invert: low overlap = high risk
        risk = 1.0 - overlap
        return round(min(max(risk, 0.0), 1.0), 3)
