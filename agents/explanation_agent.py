"""
agents/explanation_agent.py
============================
Concept Explanation Agent.

Takes a user-specified concept or term and explains it in simple,
accessible language using context retrieved from the academic document.

Goal: bridge the gap between expert content and general readers.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag.generator import RAGGenerator, GenerationResult
from config.prompts import SYSTEM_EXPLAINER, EXPLANATION_PROMPT

logger = logging.getLogger(__name__)


class ExplanationAgent(BaseAgent):
    """
    Explains academic concepts in simple terms using document context.

    Usage:
        agent = ExplanationAgent()
        result = agent("attention mechanism", doc_id="transformer_paper")
        print(result.answer)
    """

    def __init__(self, generator: Optional[RAGGenerator] = None):
        super().__init__(generator)

    @property
    def agent_name(self) -> str:
        return "Explanation Agent"

    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Args:
            input_data: The concept or term to explain
            doc_id:     Document context for the explanation
        """
        return self._generator.generate(
            query=input_data,
            system_prompt=SYSTEM_EXPLAINER,
            prompt_template=EXPLANATION_PROMPT,
            doc_id=doc_id,
            top_k=6,
            chunk_types=["text"],
            extra_variables={"concept": input_data},
        )
