"""
evaluation/qa_eval.py
======================
QA Evaluation Module — Exact Match (EM) and Token-level F1.

Standard metrics from the SQuAD benchmark:
  - Exact Match (EM): 1 if predicted answer == reference answer (normalized), else 0
  - Token F1: token-level precision/recall/F1 between predicted and reference

Normalization:
  - Lowercase
  - Remove punctuation
  - Remove articles (a, an, the)
  - Normalize whitespace

Reference: Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine
           Comprehension of Text. EMNLP 2016.
"""

import logging
import re
import string
from dataclasses import dataclass
from typing import Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Normalize text for fair comparison:
    lowercase → remove punctuation → remove articles → normalize whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove leading articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# EXACT MATCH
# ─────────────────────────────────────────────

def exact_match(prediction: str, references: Union[str, list[str]]) -> float:
    """
    Compute exact match score.

    Args:
        prediction: Model-generated answer
        references: Ground-truth answer(s). If multiple, returns 1 if any matches.

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if isinstance(references, str):
        references = [references]
    pred_norm = _normalize(prediction)
    return float(any(_normalize(ref) == pred_norm for ref in references))


# ─────────────────────────────────────────────
# TOKEN-LEVEL F1
# ─────────────────────────────────────────────

def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """
    Compute token-level Precision, Recall, and F1.

    Returns:
        (precision, recall, f1) as floats in [0, 1]
    """
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    # Count common tokens (handle duplicates with multiset intersection)
    from collections import Counter
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def best_token_f1(
    prediction: str,
    references: Union[str, list[str]],
) -> tuple[float, float, float]:
    """
    Compute the best token F1 across multiple reference answers.

    Returns:
        (precision, recall, f1) for the best-matching reference
    """
    if isinstance(references, str):
        references = [references]
    best = max(token_f1(prediction, ref) for ref in references)
    return best


# ─────────────────────────────────────────────
# EVALUATOR CLASS
# ─────────────────────────────────────────────

@dataclass
class QAEvalResult:
    """Result of QA evaluation on a single example."""
    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float

    def summary(self) -> str:
        return (
            f"EM={self.exact_match:.3f} | "
            f"F1={self.token_f1:.3f} "
            f"(P={self.token_precision:.3f}, R={self.token_recall:.3f})"
        )


class QAEvaluator:
    """
    Evaluates QA answers using Exact Match and Token F1.

    Usage:
        evaluator = QAEvaluator()
        result = evaluator.evaluate("Paris", "Paris is the capital of France")
        print(result.summary())
    """

    def evaluate(
        self,
        prediction: str,
        references: Union[str, list[str]],
    ) -> QAEvalResult:
        """
        Evaluate a single prediction against reference(s).

        Args:
            prediction: Generated answer string
            references: One or more ground-truth answers

        Returns:
            QAEvalResult
        """
        em = exact_match(prediction, references)
        p, r, f = best_token_f1(prediction, references)
        result = QAEvalResult(
            exact_match=em,
            token_precision=p,
            token_recall=r,
            token_f1=f,
        )
        logger.info(f"QA Eval: {result.summary()}")
        return result

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[Union[str, list[str]]],
    ) -> dict[str, float]:
        """
        Evaluate a batch and return macro-averaged metrics.

        Returns:
            dict with keys: avg_em, avg_f1, avg_precision, avg_recall
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        results = [self.evaluate(p, r) for p, r in zip(predictions, references)]
        n = len(results)
        return {
            "avg_em":        round(sum(r.exact_match for r in results) / n, 4),
            "avg_f1":        round(sum(r.token_f1 for r in results) / n, 4),
            "avg_precision": round(sum(r.token_precision for r in results) / n, 4),
            "avg_recall":    round(sum(r.token_recall for r in results) / n, 4),
            "num_examples":  n,
        }
