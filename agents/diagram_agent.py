"""
agents/diagram_agent.py
========================
Diagram / Image Explanation Agent.

Given an image caption (from BLIP) and contextual text from surrounding pages,
generates a detailed academic description of what the diagram shows and how
it relates to the document's academic content.
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag.generator import RAGGenerator, GenerationResult
from config.prompts import SYSTEM_DIAGRAM, DIAGRAM_PROMPT

logger = logging.getLogger(__name__)


class DiagramAgent(BaseAgent):
    """
    Analyzes and explains diagrams/figures extracted from academic PDFs.

    Usage:
        agent = DiagramAgent()
        result = agent(
            "a diagram showing neural network layers",
            doc_id="dl_paper",
            caption="neural network architecture diagram with input, hidden, and output layers",
            page=5,
        )
        print(result.answer)
    """

    def __init__(self, generator: Optional[RAGGenerator] = None):
        super().__init__(generator)

    @property
    def agent_name(self) -> str:
        return "Diagram Agent"

    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        caption: str = "",
        page: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Args:
            input_data: General description / query about the diagram
            doc_id:     Document scope
            caption:    BLIP-generated caption for the image
            page:       Page number (used to retrieve nearby text context)
        """
        # Build a retrieval query using the caption + user description
        effective_caption = caption or input_data
        retrieval_query = f"figure diagram image {effective_caption}"
        if page:
            retrieval_query += f" page {page}"

        return self._generator.generate(
            query=retrieval_query,
            system_prompt=SYSTEM_DIAGRAM,
            prompt_template=DIAGRAM_PROMPT,
            doc_id=doc_id,
            top_k=5,
            chunk_types=["text", "image"],
            extra_variables={"caption": effective_caption},
        )
