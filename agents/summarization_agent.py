"""
agents/summarization_agent.py
==============================
Document Summarization Agent.

Generates a structured academic summary covering:
  Objective → Methodology → Key Findings → Conclusions → Key Terms

Strategy:
  - Retrieves a wide set of chunks (higher top_k) to cover the full document
  - Uses a dedicated summarization prompt
  - Because summarization needs breadth, it uses a larger top_k than QA
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag.generator import RAGGenerator, GenerationResult
from config.prompts import SYSTEM_SUMMARIZER, SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)

# Summarization needs more context than a point question
SUMMARIZATION_TOP_K = 15


class SummarizationAgent(BaseAgent):
    """
    Produces a structured academic summary of a document.

    Usage:
        agent = SummarizationAgent()
        result = agent("Summarize this document", doc_id="paper")
        print(result.answer)
    """

    def __init__(
        self,
        generator: Optional[RAGGenerator] = None,
        top_k: int = SUMMARIZATION_TOP_K,
    ):
        super().__init__(generator)
        self.top_k = top_k

    @property
    def agent_name(self) -> str:
        return "Summarization Agent"

    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Args:
            input_data: Ignored (summarization is not query-dependent)
            doc_id:     Document to summarize (highly recommended to set this)
        """
        # Generic query that maximizes document coverage
        summary_query = (
            "main objective contribution methodology results conclusions key findings"
        )

        return self._generator.generate(
            query=summary_query,
            system_prompt=SYSTEM_SUMMARIZER,
            prompt_template=SUMMARIZATION_PROMPT,
            doc_id=doc_id,
            top_k=self.top_k,
            chunk_types=["text", "table"],
            score_threshold=0.0,
        )
