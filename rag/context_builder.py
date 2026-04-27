"""
rag/context_builder.py
=======================
Builds a structured, source-grounded context string from retrieved chunks.

Responsibilities:
  - Deduplicate retrieved chunks
  - Format context with source citations (Page X, Section Y)
  - Truncate to fit within the LLM's context window
  - Produce a separate sources list for display in the UI
  - Compute a basic context coverage confidence score
"""

import logging
from typing import Any

from ingestion.chunker import count_tokens
from config.settings import LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# Maximum tokens reserved for the context portion of the prompt
MAX_CONTEXT_TOKENS = LLM_MAX_TOKENS * 3   # generous context budget


class ContextBuilder:
    """
    Assembles retrieved chunks into a formatted context block
    suitable for injection into LLM prompts.

    Usage:
        builder = ContextBuilder()
        context_str, sources, confidence = builder.build(retrieved_chunks)
    """

    def __init__(self, max_context_tokens: int = MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        chunks: list[dict[str, Any]],
        include_scores: bool = False,
    ) -> tuple[str, list[dict], float]:
        """
        Build context from a list of retrieved chunk dicts.

        Args:
            chunks:         list of retrieval result dicts
            include_scores: whether to include similarity scores in context header

        Returns:
            (context_string, sources_list, confidence_score)

            context_string: formatted text ready for LLM prompt injection
            sources_list:   list of {page, section, doc_id, score} dicts
            confidence_score: float 0–1 based on top chunk scores
        """
        if not chunks:
            return ("No relevant content found in the document.", [], 0.0)

        # Deduplicate: keep highest-scoring chunk for identical texts
        seen_texts: set[str] = set()
        deduped: list[dict] = []
        for chunk in sorted(chunks, key=lambda x: x.get("score", 0), reverse=True):
            text = chunk.get("text", "").strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                deduped.append(chunk)

        # Build formatted context, respecting token budget
        context_parts: list[str] = []
        sources: list[dict] = []
        total_tokens = 0

        for i, chunk in enumerate(deduped, start=1):
            text = chunk.get("text", "").strip()
            page = chunk.get("page", "?")
            section = chunk.get("section", "")
            doc_id = chunk.get("doc_id", "")
            score = chunk.get("score", 0.0)
            chunk_type = chunk.get("type", "text")

            # Citation header for this chunk
            citation = f"[Source {i}] Page {page}"
            if section:
                citation += f" | Section: {section}"
            if include_scores:
                citation += f" | Score: {score:.3f}"
            if chunk_type != "text":
                citation += f" | Type: {chunk_type.upper()}"

            chunk_text = f"{citation}\n{text}"
            chunk_tokens = count_tokens(chunk_text)

            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.debug(
                    f"Context budget reached at chunk {i}; "
                    f"stopped at {total_tokens} tokens"
                )
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            sources.append({
                "citation": f"Source {i}",
                "page": page,
                "section": section,
                "doc_id": doc_id,
                "score": score,
                "type": chunk_type,
            })

        context_string = "\n\n---\n\n".join(context_parts)

        # Confidence score: weighted average of top chunk similarities
        confidence = self._compute_confidence(sources)
        logger.debug(
            f"Context built: {len(context_parts)} chunks, "
            f"{total_tokens} tokens, confidence={confidence:.2f}"
        )
        return context_string, sources, confidence

    def build_sources_citation(self, sources: list[dict]) -> str:
        """
        Build a human-readable citation string from the sources list.

        Example: "[Source 1] Page 3 | Section: Introduction"
        """
        if not sources:
            return "No sources found."
        lines = []
        for src in sources:
            line = f"**{src['citation']}** — Page {src['page']}"
            if src.get("section"):
                line += f" | *{src['section']}*"
            if src.get("type", "text") != "text":
                line += f" [{src['type'].upper()}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _compute_confidence(sources: list[dict]) -> float:
        """
        Compute an overall context confidence score (0–1).

        Uses the mean of the top-3 similarity scores.
        A score above 0.7 = high confidence, 0.4–0.7 = medium, < 0.4 = low.
        """
        if not sources:
            return 0.0
        scores = [s.get("score", 0.0) for s in sources[:3]]
        return round(sum(scores) / len(scores), 3)

    @staticmethod
    def confidence_label(score: float) -> str:
        """Map a confidence score to a human-readable label."""
        if score >= 0.70:
            return "High"
        elif score >= 0.45:
            return "Medium"
        else:
            return "Low"
