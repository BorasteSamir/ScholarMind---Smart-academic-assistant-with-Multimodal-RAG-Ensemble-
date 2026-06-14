"""
rag/generator.py
=================
RAG generator — now delegates the full pipeline to the LangGraph graph
defined in graph.py.

Public interface is IDENTICAL to before:
  - GenerationResult dataclass (unchanged)
  - RAGGenerator.generate(query, system_prompt, prompt_template, …) → GenerationResult
  - get_generator() singleton

The LangGraph graph handles: retrieve → build_context → ensemble_generate
  → hallucination_check → (optional retry) → format_result.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT DATA CLASS  (unchanged — app.py depends on this)
# ─────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    citations_text: str = ""
    confidence_score: float = 0.0
    confidence_label: str = "Low"
    hallucination_flag: bool = False
    hallucination_warning: str = ""
    generation_time_sec: float = 0.0
    model: str = ""
    retrieved_chunks: list[dict] = field(default_factory=list)
    ensemble_result: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "citations_text": self.citations_text,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "hallucination_flag": self.hallucination_flag,
            "hallucination_warning": self.hallucination_warning,
            "generation_time_sec": self.generation_time_sec,
            "model": self.model,
        }


# ─────────────────────────────────────────────
# INTENT INFERENCE  (maps agent-specific call params → intent string)
# ─────────────────────────────────────────────

# Maps the system_prompt object to its intent so RAGGenerator.generate()
# (called by each agent's _run()) can pass the right intent to the graph.
from config.prompts import (
    SYSTEM_QA, SYSTEM_SUMMARIZER, SYSTEM_EXPLAINER,
    SYSTEM_FORMULA, SYSTEM_DIAGRAM,
)

_SYSTEM_TO_INTENT: dict[str, str] = {
    SYSTEM_QA:        "qa",
    SYSTEM_SUMMARIZER: "summarize",
    SYSTEM_EXPLAINER:  "explain",
    SYSTEM_FORMULA:    "formula",
    SYSTEM_DIAGRAM:    "diagram",
}


# ─────────────────────────────────────────────
# RAG GENERATOR
# ─────────────────────────────────────────────

class RAGGenerator:
    """
    Thin orchestrator that converts agent-style generate() calls into
    LangGraph run_graph() calls and wraps the result in GenerationResult.
    """

    def generate(
        self,
        query: str,
        system_prompt: str,
        prompt_template: str,
        doc_id: Optional[str] = None,
        top_k: int = RETRIEVAL_TOP_K,
        chunk_types: Optional[list[str]] = None,
        extra_variables: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> GenerationResult:
        # Lazy import avoids circular deps at module load time
        from graph import run_graph

        intent = _SYSTEM_TO_INTENT.get(system_prompt, "qa")

        # Merge caller's extra variables so graph nodes can see them
        merged_extra = dict(extra_variables or {})

        try:
            state = run_graph(
                query=query,
                intent=intent,
                doc_id=doc_id,
                extra_variables=merged_extra,
            )

            # Wrap EnsembleResult-like dict in a simple object so app.py
            # can still access .best_model, .scores, .models, etc.
            ensemble_obj = None
            if state.get("ensemble_result"):
                ensemble_obj = _EnsembleResultView(state["ensemble_result"])

            return GenerationResult(
                answer=state["answer"],
                sources=state["sources"],
                citations_text=state["citations_text"],
                confidence_score=state["confidence_score"],
                confidence_label=state["confidence_label"],
                hallucination_flag=state["hallucination_flag"],
                hallucination_warning=state["hallucination_warning"],
                generation_time_sec=state["generation_time_sec"],
                model=state["model"],
                retrieved_chunks=state["chunks"],
                ensemble_result=ensemble_obj,
            )

        except Exception as exc:
            logger.error(f"[RAGGenerator] graph invocation failed: {exc}")
            return GenerationResult(
                answer=f"⚠️ Generation failed: {exc}",
                confidence_score=0.0,
                confidence_label="Low",
                generation_time_sec=0.0,
            )


class _EnsembleResultView:
    """
    Thin view that exposes the ensemble result dict as attributes,
    exactly matching the EnsembleResult interface used in app.py.
    """
    def __init__(self, d: dict):
        self.final_answer  = d.get("final_answer", "")
        self.confidence    = d.get("confidence", 0.0)
        self.best_model    = d.get("best_model", "")
        self.all_responses = d.get("all_responses", [])
        self.scores        = d.get("scores", [])
        self.models        = d.get("models", [])

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.70:
            return "High"
        elif self.confidence >= 0.45:
            return "Medium"
        return "Low"

    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "best_model": self.best_model,
            "all_responses": self.all_responses,
            "scores": self.scores,
            "models": self.models,
        }


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_default_generator: Optional[RAGGenerator] = None


def get_generator() -> RAGGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = RAGGenerator()
    return _default_generator
