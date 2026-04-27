"""
agents/formula_agent.py
========================
Mathematical Formula Explanation Agent.

Given a LaTeX formula (or raw formula text), retrieves surrounding document
context and generates a step-by-step explanation covering:
  - What the formula represents
  - Breakdown of each symbol/variable
  - Units and dimensions
  - Practical significance in the paper
  - Intuitive (non-mathematical) interpretation
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag.generator import RAGGenerator, GenerationResult
from config.prompts import SYSTEM_FORMULA, FORMULA_PROMPT

logger = logging.getLogger(__name__)


class FormulaAgent(BaseAgent):
    """
    Explains mathematical formulas found in academic documents.

    Usage:
        agent = FormulaAgent()
        result = agent(
            r"E = mc^2",
            doc_id="physics_paper",
            latex=r"E = mc^2"
        )
        print(result.answer)
    """

    def __init__(self, generator: Optional[RAGGenerator] = None):
        super().__init__(generator)

    @property
    def agent_name(self) -> str:
        return "Formula Agent"

    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        latex: str = "",
        **kwargs,
    ) -> GenerationResult:
        """
        Args:
            input_data: Human-readable formula description or raw formula text
            doc_id:     Document context for retrieval
            latex:      LaTeX representation of the formula (preferred)
        """
        formula_latex = latex or input_data
        # Query retrieves text near this formula in the document
        retrieval_query = f"formula equation {input_data} mathematical expression"

        return self._generator.generate(
            query=retrieval_query,
            system_prompt=SYSTEM_FORMULA,
            prompt_template=FORMULA_PROMPT,
            doc_id=doc_id,
            top_k=5,
            chunk_types=["text", "formula"],
            extra_variables={"formula_latex": formula_latex},
        )
