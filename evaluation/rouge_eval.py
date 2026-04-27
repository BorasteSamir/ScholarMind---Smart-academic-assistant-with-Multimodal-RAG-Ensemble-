"""
evaluation/rouge_eval.py
=========================
ROUGE-based evaluation for summarization quality.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
n-gram overlap between generated summaries and reference summaries.

Metrics computed:
  - ROUGE-1: Unigram overlap (individual words)
  - ROUGE-2: Bigram overlap (word pairs)
  - ROUGE-L: Longest Common Subsequence (sentence-level structure)

Each metric returns:
  - Precision: how much of the generated text is in the reference
  - Recall:    how much of the reference is in the generated text
  - F1:        harmonic mean of precision and recall

Uses the `rouge-score` library from Google.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class RougeScore:
    """Container for ROUGE evaluation results."""
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    rouge2_precision: float
    rouge2_recall: float
    rouge2_f1: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float

    def summary(self) -> str:
        return (
            f"ROUGE-1: P={self.rouge1_precision:.3f} R={self.rouge1_recall:.3f} "
            f"F1={self.rouge1_f1:.3f}\n"
            f"ROUGE-2: P={self.rouge2_precision:.3f} R={self.rouge2_recall:.3f} "
            f"F1={self.rouge2_f1:.3f}\n"
            f"ROUGE-L: P={self.rougeL_precision:.3f} R={self.rougeL_recall:.3f} "
            f"F1={self.rougeL_f1:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "rouge1": {
                "precision": self.rouge1_precision,
                "recall": self.rouge1_recall,
                "f1": self.rouge1_f1,
            },
            "rouge2": {
                "precision": self.rouge2_precision,
                "recall": self.rouge2_recall,
                "f1": self.rouge2_f1,
            },
            "rougeL": {
                "precision": self.rougeL_precision,
                "recall": self.rougeL_recall,
                "f1": self.rougeL_f1,
            },
        }


# ─────────────────────────────────────────────
# ROUGE EVALUATOR
# ─────────────────────────────────────────────

class RougeEvaluator:
    """
    Computes ROUGE scores for summarization evaluation.

    Usage:
        evaluator = RougeEvaluator()
        scores = evaluator.evaluate(generated_summary, reference_summary)
        print(scores.summary())
    """

    def __init__(self, use_stemmer: bool = True):
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=use_stemmer,
        )

    def evaluate(self, generated: str, reference: str) -> RougeScore:
        """
        Compute ROUGE scores between a generated and reference text.

        Args:
            generated:  Text produced by the summarization agent
            reference:  Ground-truth reference summary

        Returns:
            RougeScore dataclass
        """
        if not generated.strip() or not reference.strip():
            logger.warning("Empty generated or reference text for ROUGE evaluation")
            return RougeScore(0, 0, 0, 0, 0, 0, 0, 0, 0)

        scores = self._scorer.score(reference, generated)

        result = RougeScore(
            rouge1_precision=scores["rouge1"].precision,
            rouge1_recall=scores["rouge1"].recall,
            rouge1_f1=scores["rouge1"].fmeasure,
            rouge2_precision=scores["rouge2"].precision,
            rouge2_recall=scores["rouge2"].recall,
            rouge2_f1=scores["rouge2"].fmeasure,
            rougeL_precision=scores["rougeL"].precision,
            rougeL_recall=scores["rougeL"].recall,
            rougeL_f1=scores["rougeL"].fmeasure,
        )
        logger.info(f"ROUGE evaluation:\n{result.summary()}")
        return result

    def evaluate_batch(
        self,
        generated_list: list[str],
        reference_list: list[str],
    ) -> list[RougeScore]:
        """Evaluate multiple (generated, reference) pairs."""
        if len(generated_list) != len(reference_list):
            raise ValueError("Generated and reference lists must have the same length")
        return [
            self.evaluate(gen, ref)
            for gen, ref in zip(generated_list, reference_list)
        ]

    def average_scores(self, scores: list[RougeScore]) -> RougeScore:
        """Compute macro-average ROUGE scores over a list of RougeScore objects."""
        n = len(scores)
        if n == 0:
            return RougeScore(0, 0, 0, 0, 0, 0, 0, 0, 0)
        return RougeScore(
            rouge1_precision=sum(s.rouge1_precision for s in scores) / n,
            rouge1_recall=sum(s.rouge1_recall for s in scores) / n,
            rouge1_f1=sum(s.rouge1_f1 for s in scores) / n,
            rouge2_precision=sum(s.rouge2_precision for s in scores) / n,
            rouge2_recall=sum(s.rouge2_recall for s in scores) / n,
            rouge2_f1=sum(s.rouge2_f1 for s in scores) / n,
            rougeL_precision=sum(s.rougeL_precision for s in scores) / n,
            rougeL_recall=sum(s.rougeL_recall for s in scores) / n,
            rougeL_f1=sum(s.rougeL_f1 for s in scores) / n,
        )
