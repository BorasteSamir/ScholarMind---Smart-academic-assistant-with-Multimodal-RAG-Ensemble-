"""
agents/qa_agent.py
===================
Question-Answering Agent.

Retrieves relevant context from the vector store and generates a precise,
source-grounded answer to the user's question.

Key behaviors:
  - Uses semantic retrieval across all chunk types (text, table)
  - Injects sources as [Page X] citations in the answer
  - Appends a confidence rating
  - Flags potential hallucinations
"""

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag.generator import RAGGenerator, GenerationResult
from config.prompts import SYSTEM_QA, QA_PROMPT
from config.settings import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


class QAAgent(BaseAgent):
    """
    Answers factual questions about an academic document using RAG.

    Usage:
        agent = QAAgent()
        result = agent("What is the main contribution of this paper?", doc_id="paper")
        print(result.answer)
        print(result.citations_text)
    """

    def __init__(self, generator: Optional[RAGGenerator] = None, top_k: int = RETRIEVAL_TOP_K):
        super().__init__(generator)
        self.top_k = top_k

    @property
    def agent_name(self) -> str:
        return "QA Agent"

    def _run(
        self,
        input_data: str,
        doc_id: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Args:
            input_data: The question to answer
            doc_id:     Restrict retrieval to a specific document
        """
        return self._generator.generate(
            query=input_data,
            system_prompt=SYSTEM_QA,
            prompt_template=QA_PROMPT,
            doc_id=doc_id,
            top_k=self.top_k,
            chunk_types=["text", "table"],
            extra_variables={"question": input_data},
        )
