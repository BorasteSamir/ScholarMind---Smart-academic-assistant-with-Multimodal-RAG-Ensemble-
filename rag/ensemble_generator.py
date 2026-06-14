"""
rag/ensemble_generator.py
==========================
Backward-compatibility shim.

The ensemble logic now lives in graph.py (_run_ensemble_sync).
This module re-exports EnsembleResult and ensemble_generate_sync so that
any code importing from here continues to work without changes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from graph import _run_ensemble_sync


@dataclass
class EnsembleResult:
    final_answer:  str
    confidence:    float
    best_model:    str
    all_responses: list[str]   = field(default_factory=list)
    scores:        list[float] = field(default_factory=list)
    models:        list[str]   = field(default_factory=list)
    elapsed_sec:   float       = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer":  self.final_answer,
            "confidence":    self.confidence,
            "best_model":    self.best_model,
            "all_responses": self.all_responses,
            "scores":        self.scores,
            "models":        self.models,
            "elapsed_sec":   self.elapsed_sec,
        }

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.70:
            return "High"
        elif self.confidence >= 0.45:
            return "Medium"
        return "Low"


def ensemble_generate_sync(
    query: str,
    retrieved_chunks: list[dict],
    prompt_template: str,
) -> EnsembleResult:
    """Delegates to graph._run_ensemble_sync and wraps in EnsembleResult."""
    d = _run_ensemble_sync(query, retrieved_chunks, prompt_template)
    return EnsembleResult(
        final_answer=d["final_answer"],
        confidence=d["confidence"],
        best_model=d["best_model"],
        all_responses=d["all_responses"],
        scores=d["scores"],
        models=d["models"],
    )
