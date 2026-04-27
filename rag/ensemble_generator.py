"""
rag/ensemble_generator.py
==========================
Multi-model bagging ensemble for RAG generation.

Architecture:
  - 5 Groq models called in parallel via asyncio
  - Each model receives different context (top_k variation) and prompt style
  - Each response is scored on 4 axes:
      0.4 * token_overlap(response, context)
      0.3 * (1 - hallucination_risk)
      0.2 * semantic_similarity(response, context)
      0.1 * answer_quality(response)
  - Best scoring response is selected as final answer
  - Confidence = mean pairwise embedding similarity across all 5 responses

Output:
  {
    "final_answer": str,
    "confidence": float,
    "best_model": str,
    "all_responses": list[str],
    "scores": list[float]
  }
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from config.settings import GROQ_API_KEY, GROQ_MODEL
from evaluation.hallucination_detector import _token_overlap_ratio

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENSEMBLE CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    model:       str
    temperature: float
    top_p:       float
    top_k:       int
    system_prompt: str


ENSEMBLE_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        top_p=0.90,
        top_k=3,
        system_prompt=(
            "You are an academic assistant. "
            "Answer strictly from the provided context. Be concise and precise. "
            "Cite page numbers using [Page X] notation. "
            "If the answer is not in the context, say so explicitly."
        ),
    ),
    ModelConfig(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        top_p=0.95,
        top_k=5,
        system_prompt=(
            "You are an academic assistant. "
            "Explain step-by-step using only the provided context. "
            "Structure your answer clearly with numbered steps where applicable. "
            "Always cite sources."
        ),
    ),
    ModelConfig(
        model="qwen/qwen3-32b",
        temperature=0.6,
        top_p=0.85,
        top_k=7,
        system_prompt=(
            "You are an academic assistant. "
            "Provide a structured bullet-point answer based solely on the context. "
            "Use '•' for main points and '  -' for sub-points. "
            "Include page citations."
        ),
    ),
    ModelConfig(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        top_p=0.90,
        top_k=5,
        system_prompt=(
            "You are an academic assistant. "
            "Focus on accuracy. Avoid assumptions or external knowledge. "
            "Answer only what the context supports. "
            "Flag any uncertainty explicitly."
        ),
    ),
    ModelConfig(
        model="qwen/qwen3-32b",
        temperature=0.5,
        top_p=0.80,
        top_k=5,
        system_prompt=(
            "You are an academic assistant. "
            "Give a detailed explanation with concrete examples drawn from the context. "
            "Be thorough but stay grounded in the provided material."
        ),
    ),
]


# ─────────────────────────────────────────────
# CONTEXT VARIATION HELPERS
# ─────────────────────────────────────────────

def _build_context_from_chunks(chunks: list[dict]) -> str:
    """Format a list of retrieved chunks into a context string."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        header = f"[Source {i}] Page {page}"
        if section:
            header += f" | {section}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "No context available."


def _summarize_context(chunks: list[dict], max_chars: int = 1500) -> str:
    """
    Lightweight extractive summarization of context:
    takes the first sentence of each chunk up to max_chars.
    """
    sentences = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        # Take first sentence
        first = re.split(r"(?<=[.!?])\s+", text)
        if first:
            sentences.append(f"[Page {chunk.get('page','?')}] {first[0]}")
    summary = " ".join(sentences)
    return summary[:max_chars] if len(summary) > max_chars else summary


def _prepare_context(
    chunks: list[dict],
    config_index: int,
    all_chunks: list[dict],
) -> str:
    """
    Return context string with variation per model index:
      0 → top 3 chunks
      1 → top 5 chunks
      2 → top 7 chunks
      3 → shuffled chunks (top 5)
      4 → summarized context (top 5)
    """
    if config_index == 0:
        return _build_context_from_chunks(all_chunks[:3])
    elif config_index == 1:
        return _build_context_from_chunks(all_chunks[:5])
    elif config_index == 2:
        return _build_context_from_chunks(all_chunks[:7])
    elif config_index == 3:
        shuffled = all_chunks[:5][:]
        random.shuffle(shuffled)
        return _build_context_from_chunks(shuffled)
    else:
        return _summarize_context(all_chunks[:5])


# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────

def _score_token_overlap(response: str, context: str) -> float:
    """Reuse existing hallucination detector's overlap ratio."""
    return _token_overlap_ratio(response, context)


def _score_hallucination(response: str, context: str) -> float:
    """
    Returns hallucination SAFETY score (higher = safer).
    1.0 - risk_score so it can be maximized like other scores.
    """
    from evaluation.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()
    risk = detector.score(context, response)
    return round(1.0 - risk, 4)


def _score_semantic_similarity(response: str, context: str) -> float:
    """
    Cosine similarity between response and context embeddings.
    Uses the existing singleton embedder.
    """
    try:
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        resp_vec = embedder.embed(response)
        ctx_vec = embedder.embed(context[:2000])  # cap context length
        # Both are already L2-normalized → dot product = cosine similarity
        similarity = float(np.dot(resp_vec, ctx_vec))
        return round(max(0.0, min(1.0, similarity)), 4)
    except Exception as exc:
        logger.warning(f"Semantic similarity scoring failed: {exc}")
        return 0.0


def _score_answer_quality(response: str) -> float:
    """
    Heuristic answer quality score based on:
      - Length (too short or too long penalized)
      - Presence of citations [Page X]
      - Absence of refusal phrases
    """
    score = 0.0
    length = len(response.strip())

    # Length score: ideal 100–800 chars
    if 100 <= length <= 800:
        score += 0.5
    elif 50 <= length < 100 or 800 < length <= 1500:
        score += 0.3
    elif length > 1500:
        score += 0.2
    else:
        score += 0.0

    # Citation presence
    if re.search(r"\[Page \d+\]", response):
        score += 0.3

    # Penalize refusal / uncertainty phrases
    refusal_patterns = [
        r"i (don't|do not|cannot|can't) (know|find|answer)",
        r"not (found|available|present) in (the )?context",
        r"no (relevant|sufficient) (information|context)",
    ]
    if any(re.search(p, response.lower()) for p in refusal_patterns):
        score -= 0.2

    return round(max(0.0, min(1.0, score)), 4)


def compute_score(response: str, context: str) -> float:
    """
    Composite score:
      0.4 * token_overlap
      0.3 * hallucination_safety
      0.2 * semantic_similarity
      0.1 * answer_quality
    """
    w_overlap    = 0.4
    w_halluc     = 0.3
    w_semantic   = 0.2
    w_quality    = 0.1

    overlap   = _score_token_overlap(response, context)
    halluc    = _score_hallucination(response, context)
    semantic  = _score_semantic_similarity(response, context)
    quality   = _score_answer_quality(response)

    total = (
        w_overlap  * overlap  +
        w_halluc   * halluc   +
        w_semantic * semantic +
        w_quality  * quality
    )
    logger.debug(
        f"Score breakdown → overlap={overlap:.3f} halluc={halluc:.3f} "
        f"semantic={semantic:.3f} quality={quality:.3f} → total={total:.3f}"
    )
    return round(total, 4)


# ─────────────────────────────────────────────
# CONFIDENCE COMPUTATION
# ─────────────────────────────────────────────

def compute_confidence(responses: list[str]) -> float:
    """
    Mean pairwise cosine similarity across all response embeddings.
    High similarity → models agree → high confidence.
    """
    try:
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        vecs = np.array([embedder.embed(r) for r in responses])  # (5, 384)

        # Pairwise dot products (already normalized)
        sim_matrix = vecs @ vecs.T  # (5, 5)
        n = len(responses)
        # Sum off-diagonal elements
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(sim_matrix[i, j])
                count += 1
        confidence = total / count if count > 0 else 0.0
        return round(max(0.0, min(1.0, confidence)), 4)
    except Exception as exc:
        logger.warning(f"Confidence computation failed: {exc}")
        return 0.5


# ─────────────────────────────────────────────
# SINGLE MODEL ASYNC CALL
# ─────────────────────────────────────────────

async def _call_model(
    config: ModelConfig,
    context: str,
    query: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Call a single Groq model asynchronously.
    Returns dict with model, response, context used, and error if any.
    """
    async with semaphore:
        start = time.time()
        try:
            # Build user prompt from template
            user_prompt = prompt_template.format(
                context=context,
                question=query,
                concept=query,
                sources="See context above.",
            )

            # Run blocking Groq call in thread pool to not block event loop
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None,
                _groq_call_sync,
                config,
                user_prompt,
            )

            elapsed = round(time.time() - start, 2)
            logger.info(
                f"[Ensemble] Model={config.model} | "
                f"Time={elapsed}s | "
                f"Response length={len(response_text)} chars"
            )
            return {
                "model": config.model,
                "response": response_text,
                "context": context,
                "error": None,
                "elapsed": elapsed,
            }

        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            logger.error(f"[Ensemble] Model={config.model} failed: {exc}")
            return {
                "model": config.model,
                "response": "",
                "context": context,
                "error": str(exc),
                "elapsed": elapsed,
            }


def _groq_call_sync(config: ModelConfig, user_prompt: str) -> str:
    """Synchronous Groq API call (runs in thread executor)."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# ENSEMBLE RESULT
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# MAIN ENSEMBLE FUNCTION
# ─────────────────────────────────────────────

async def ensemble_generate(
    query: str,
    retrieved_chunks: list[dict],
    prompt_template: str,
    max_concurrent: int = 5,
) -> EnsembleResult:
    """
    Run 5 Groq models in parallel, score each response, return the best.

    Args:
        query:            User query string
        retrieved_chunks: Top-k chunks from ChromaDB retriever
        prompt_template:  Prompt template string with {context}, {question} placeholders
        max_concurrent:   Max parallel API calls (default 5)

    Returns:
        EnsembleResult with final_answer, confidence, scores, etc.
    """
    start_total = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build context variation per model
    tasks = []
    for i, config in enumerate(ENSEMBLE_CONFIGS):
        context = _prepare_context(retrieved_chunks, i, retrieved_chunks)
        tasks.append(
            _call_model(config, context, query, prompt_template, semaphore)
        )

    logger.info(f"[Ensemble] Launching {len(tasks)} models in parallel for query: '{query[:60]}'")

    # Run all models in parallel
    raw_results: list[dict] = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter out failed calls
    valid_results = [r for r in raw_results if r["response"] and not r["error"]]

    if not valid_results:
        logger.error("[Ensemble] All models failed. Returning fallback response.")
        return EnsembleResult(
            final_answer="⚠️ All ensemble models failed to generate a response. Please try again.",
            confidence=0.0,
            best_model="none",
            all_responses=[],
            scores=[],
            models=[],
            elapsed_sec=round(time.time() - start_total, 2),
        )

    # Score each valid response
    scores = []
    for r in valid_results:
        score = compute_score(r["response"], r["context"])
        scores.append(score)
        logger.info(
            f"[Ensemble] Model={r['model']} | Score={score:.4f} | "
            f"Response preview: {r['response'][:80]}..."
        )

    # Select best response
    best_idx = int(np.argmax(scores))
    best_result = valid_results[best_idx]

    # Compute inter-model confidence
    all_responses = [r["response"] for r in valid_results]
    confidence = compute_confidence(all_responses)

    elapsed_total = round(time.time() - start_total, 2)

    logger.info(
        f"[Ensemble] Best model={best_result['model']} | "
        f"Best score={scores[best_idx]:.4f} | "
        f"Confidence={confidence:.4f} | "
        f"Total time={elapsed_total}s"
    )

    return EnsembleResult(
        final_answer=best_result["response"],
        confidence=confidence,
        best_model=best_result["model"],
        all_responses=all_responses,
        scores=scores,
        models=[r["model"] for r in valid_results],
        elapsed_sec=elapsed_total,
    )


# ─────────────────────────────────────────────
# SYNC WRAPPER (for non-async callers)
# ─────────────────────────────────────────────

def ensemble_generate_sync(
    query: str,
    retrieved_chunks: list[dict],
    prompt_template: str,
) -> EnsembleResult:
    """
    Synchronous wrapper around ensemble_generate().
    Safe to call from Streamlit or any non-async context.
    Always runs the coroutine in a brand-new thread with its own
    event loop so it never conflicts with Streamlit's internal loop.
    """
    import concurrent.futures

    def _run_in_thread():
        # Each thread gets its own fresh event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                ensemble_generate(query, retrieved_chunks, prompt_template)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_in_thread)
        return future.result(timeout=120)
