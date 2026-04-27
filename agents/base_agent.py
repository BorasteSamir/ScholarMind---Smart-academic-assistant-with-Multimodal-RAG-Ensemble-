"""
agents/base_agent.py
=====================
Abstract base class for all agents in the Smart Academic Assistant.

Each agent encapsulates a specific academic NLP task:
  - QAAgent           — question answering
  - SummarizationAgent — document summarization
  - ExplanationAgent  — concept explanation
  - FormulaAgent      — formula explanation
  - DiagramAgent      — diagram / image explanation

Design pattern: Strategy + Template Method
  - __call__() is the public API
  - _run() is the agent-specific implementation (Template Method)
  - All agents share the same RAGGenerator infrastructure
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from rag.generator import RAGGenerator, GenerationResult, get_generator

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclasses must implement:
        - agent_name: str property
        - _run(input_data, doc_id, **kwargs) → GenerationResult

    The __call__() method wraps _run() with logging and timing.
    """

    def __init__(self, generator: Optional[RAGGenerator] = None):
        self._generator = generator or get_generator()

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Human-readable name for this agent."""
        ...

    @abstractmethod
    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Core agent logic.

        Args:
            input_data: The primary input (question, concept name, etc.)
            doc_id:     Document scope for retrieval (None = all docs)
            **kwargs:   Agent-specific additional parameters

        Returns:
            GenerationResult
        """
        ...

    def __call__(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Run the agent with logging and error handling.

        Args:
            input_data: Primary input text
            doc_id:     Document filter for retrieval
            **kwargs:   Forwarded to _run()

        Returns:
            GenerationResult
        """
        logger.info(f"[{self.agent_name}] Running on: '{input_data[:80]}'")
        start = time.time()
        try:
            result = self._run(input_data, doc_id=doc_id, **kwargs)
            elapsed = time.time() - start
            logger.info(
                f"[{self.agent_name}] Completed in {elapsed:.1f}s | "
                f"Confidence: {result.confidence_label}"
            )
            return result
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"[{self.agent_name}] Failed after {elapsed:.1f}s: {exc}")
            # Return a graceful failure result
            from dataclasses import dataclass
            return GenerationResult(
                answer=f"⚠️ {self.agent_name} encountered an error: {str(exc)}",
                sources=[],
                confidence_score=0.0,
                confidence_label="Low",
                generation_time_sec=round(elapsed, 2),
            )
