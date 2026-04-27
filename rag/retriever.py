"""
rag/retriever.py
=================
Semantic retrieval module — top-k similarity search over the vector store.
"""

import logging
from typing import Any, Optional

from embeddings.vector_store import VectorStore, get_vector_store
from config.settings import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ):
        self._store = vector_store or get_vector_store()
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
        chunk_types: Optional[list[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        results = self._store.search(
            query=query,
            top_k=k,
            doc_id=doc_id,
            chunk_types=chunk_types,
            score_threshold=threshold,
        )
        logger.debug(
            f"Retrieved {len(results)} chunks for query: '{query[:60]}...'"
            if len(query) > 60 else
            f"Retrieved {len(results)} chunks for query: '{query}'"
        )
        return results

    def retrieve_text(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve only text and table chunks."""
        return self.retrieve(
            query=query,
            doc_id=doc_id,
            top_k=top_k,
            chunk_types=["text", "table"],
        )

    def retrieve_images(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve only image chunks (matched by caption)."""
        return self.retrieve(
            query=query,
            doc_id=doc_id,
            top_k=top_k,
            chunk_types=["image"],
        )

    def retrieve_formulas(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve only formula chunks."""
        return self.retrieve(
            query=query,
            doc_id=doc_id,
            top_k=top_k,
            chunk_types=["formula"],
        )

    def retrieve_all_types(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, list[dict]]:
        all_results = self.retrieve(query=query, doc_id=doc_id, top_k=top_k)
        grouped: dict[str, list[dict]] = {"text": [], "image": [], "formula": [], "table": []}
        for result in all_results:
            chunk_type = result.get("type", "text")
            if chunk_type in grouped:
                grouped[chunk_type].append(result)
        return grouped


_default_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Return the module-level singleton Retriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever
