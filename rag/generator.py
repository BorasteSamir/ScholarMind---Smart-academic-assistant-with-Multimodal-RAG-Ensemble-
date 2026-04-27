"""
rag/generator.py
=================
LLM response generation with source citations and hallucination warnings.

Pipeline (per query):
  1. Retrieve top-k relevant chunks from vector store
  2. Build grounded context + source list
  3. Inject into prompt template
  4. Call ensemble of 5 Groq models in parallel
  5. Score all responses and select the best
  6. Append structured citation block
  7. Run hallucination detection heuristic
  8. Return a GenerationResult object

RAGGenerator is the central hub of the RAG pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rag.retriever import Retriever, get_retriever
from rag.context_builder import ContextBuilder
from rag.llm_client import BaseLLMClient, get_llm_client
from rag.ensemble_generator import ensemble_generate_sync, EnsembleResult
from evaluation.hallucination_detector import HallucinationDetector
from config.settings import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Structured output from the RAG generator.

    Attributes:
        answer:              LLM-generated answer text
        sources:             List of source dicts used in context
        citations_text:      Formatted citation block
        confidence_score:    Float 0–1 reflecting retrieval quality
        confidence_label:    "High" | "Medium" | "Low"
        hallucination_flag:  True if potential hallucination detected
        hallucination_warning: Explanation if flagged
        generation_time_sec: Wall-clock time for this generation
        model:               Name of the LLM model used
        retrieved_chunks:    Raw retrieval results (for debugging)
        ensemble_result:     Full EnsembleResult (scores, all responses, etc.)
    """
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
# RAG GENERATOR
# ─────────────────────────────────────────────

class RAGGenerator:
    """
    Core RAG generator that orchestrates retrieval → context → LLM.

    Usage:
        generator = RAGGenerator()
        result = generator.generate(
            query="Explain the attention mechanism",
            system_prompt=SYSTEM_QA,
            prompt_template=QA_PROMPT,
            doc_id="paper_name",
        )
        print(result.answer)
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
        detect_hallucinations: bool = True,
    ):
        self._retriever = retriever or get_retriever()
        self._llm_client_override = llm_client   # None = lazy-load on first call
        self._llm: Optional[BaseLLMClient] = llm_client
        self._context_builder = ContextBuilder()
        self._detect_hallucinations = detect_hallucinations
        self._hallucination_detector: Optional[HallucinationDetector] = None

    def _get_llm(self) -> BaseLLMClient:
        """Lazy-load the LLM client on first generate() call."""
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def _get_hallucination_detector(self) -> Optional[HallucinationDetector]:
        """Lazy-load hallucination detector on first use."""
        if self._detect_hallucinations and self._hallucination_detector is None:
            self._hallucination_detector = HallucinationDetector()
        return self._hallucination_detector

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
        """
        Full RAG pipeline: retrieve → context → generate → validate.

        Args:
            query:           User's natural-language query / topic
            system_prompt:   System instruction for the LLM
            prompt_template: f-string-style template with {context}, {sources}, etc.
            doc_id:          Restrict retrieval to a specific document
            top_k:           Number of chunks to retrieve
            chunk_types:     Restrict to specific chunk types
            extra_variables: Additional variables to fill into the prompt template
                             (e.g., {question}, {concept}, {formula_latex})

        Returns:
            GenerationResult
        """
        start = time.time()

        # ── Step 1: Retrieve ──
        chunks = self._retriever.retrieve(
            query=query,
            doc_id=doc_id,
            top_k=top_k,
            chunk_types=chunk_types,
            score_threshold=score_threshold,
        )

        # ── Step 2: Build context ──
        context_str, sources, confidence = self._context_builder.build(chunks)
        citations_text = self._context_builder.build_sources_citation(sources)

        # ── Step 3: Fill prompt template ──
        template_vars = {
            "context": context_str,
            "sources": citations_text,
            "question": query,
            "concept": query,
            **(extra_variables or {}),
        }
        try:
            user_prompt = prompt_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Prompt template missing variable: {e}; using raw template")
            user_prompt = prompt_template

        # ── Step 4: Ensemble Generate ──
        ensemble_result: Optional[EnsembleResult] = None
        try:
            ensemble_result = ensemble_generate_sync(
                query=query,
                retrieved_chunks=chunks,
                prompt_template=prompt_template,
            )
            answer = ensemble_result.final_answer
            # Use ensemble confidence if retrieval confidence is low
            if ensemble_result.confidence > confidence:
                confidence = ensemble_result.confidence
        except Exception as exc:
            logger.error(f"Ensemble generation failed, falling back to single LLM: {exc}")
            # Fallback to single LLM
            try:
                answer = self._get_llm().generate(
                    system_prompt=system_prompt,
                    user_prompt=prompt_template.format(
                        context=context_str,
                        sources=citations_text,
                        question=query,
                        concept=query,
                        **(extra_variables or {}),
                    ),
                )
            except Exception as fallback_exc:
                logger.error(f"Fallback LLM also failed: {fallback_exc}")
                answer = f"⚠️ Generation failed: {fallback_exc}"

        # ── Step 5: Hallucination check ──
        hallucination_flag = False
        hallucination_warning = ""
        detector = self._get_hallucination_detector()
        if detector and chunks:
            flag, warning = detector.check(
                context=context_str,
                answer=answer,
            )
            hallucination_flag = flag
            hallucination_warning = warning

        elapsed = time.time() - start
        conf_label = ContextBuilder.confidence_label(confidence)

        # Determine model name — prefer ensemble best_model
        model_name = (
            ensemble_result.best_model
            if ensemble_result
            else self._get_llm().model_name
        )

        result = GenerationResult(
            answer=answer,
            sources=sources,
            citations_text=citations_text,
            confidence_score=confidence,
            confidence_label=conf_label,
            hallucination_flag=hallucination_flag,
            hallucination_warning=hallucination_warning,
            generation_time_sec=round(elapsed, 2),
            model=model_name,
            retrieved_chunks=chunks,
            ensemble_result=ensemble_result,
        )

        logger.info(
            f"Generation complete in {elapsed:.1f}s | "
            f"Confidence: {conf_label} ({confidence:.2f}) | "
            f"Hallucination: {'⚠️ Yes' if hallucination_flag else 'No'}"
        )
        return result


# ─────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────

_default_generator: Optional[RAGGenerator] = None


def get_generator() -> RAGGenerator:
    """Return the module-level singleton RAGGenerator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = RAGGenerator()
    return _default_generator
