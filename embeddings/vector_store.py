"""
embeddings/vector_store.py
===========================
ChromaDB-backed vector store for the Smart Academic Assistant.

Stores text chunks (and image captions / formula text) along with rich metadata:
  - type: "text" | "image" | "formula" | "table"
  - source_page: int
  - section: str
  - doc_id: str
  - chunk_idx: int

Supports:
  - Upsert (add or update) chunks from a DocumentBundle
  - Top-k semantic similarity search
  - Filtered search by doc_id or chunk type
  - Collection reset (for re-indexing)

ChromaDB uses its own embedding function by default — we override with
our SentenceTransformer embedder for consistency with the retriever.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import (
    CHROMA_DIR,
    VECTOR_STORE_COLLECTION,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
)
from embeddings.embedder import Embedder, get_embedder
from embeddings.cache import get_cache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CHROMA EMBEDDING FUNCTION ADAPTER
# ─────────────────────────────────────────────

class SentenceTransformerEmbeddingFunction:
    """
    ChromaDB-compatible embedding function that delegates to our Embedder.
    Ensures the same model and normalization is used everywhere.
    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def name(self) -> str:
        return "sentence_transformer_embedding_function"

    def embed_query(self, input: str = "", **kwargs) -> list[float]:
        text = input or kwargs.get("query", "")
        return self._embedder.embed(text).tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._embedder.embed_batch(input)
        return embeddings.tolist()


# ─────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB vector store with metadata support.

    Usage:
        store = VectorStore()
        store.add_from_bundle(bundle)
        results = store.search("What is the attention mechanism?")
    """

    def __init__(
        self,
        collection_name: str = VECTOR_STORE_COLLECTION,
        persist_dir: Path = CHROMA_DIR,
        embedder: Optional[Embedder] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self._embedder = embedder or get_embedder()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    # ── Initialization ────────────────────────────

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            logger.info(f"Connecting to ChromaDB at: {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            emb_fn = SentenceTransformerEmbeddingFunction(self._embedder)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(
                f"Collection '{self.collection_name}' ready "
                f"({self._collection.count()} docs)"
            )
        return self._collection

    # ── Ingestion ────────────────────────────────

    def add_chunks(self, chunks: list[dict], doc_id: str) -> int:
        """
        Add a list of chunk dicts to the vector store.

        Args:
            chunks: List of dicts with 'text'/'caption'/'content', 'page', etc.
            doc_id: Unique document identifier

        Returns:
            Number of chunks added
        """
        collection = self._get_collection()
        cache = get_cache()

        ids, texts, metadatas = [], [], []
        type_counters: dict[str, int] = {}

        for chunk in chunks:
            chunk_type = chunk.get("type", "text")

            # Choose the right text field
            if chunk_type == "image":
                text = chunk.get("caption", "")
            elif chunk_type == "formula":
                text = f"{chunk.get('content', '')} {chunk.get('latex', '')}".strip()
            else:
                text = chunk.get("text", chunk.get("content", ""))

            if not text.strip():
                continue

            chunk_idx = type_counters.get(chunk_type, 0)
            type_counters[chunk_type] = chunk_idx + 1
            chunk_id = f"{doc_id}_{chunk_type}_{chunk_idx}"

            page = chunk.get("page", 0)
            # page might be a list — take the first
            if isinstance(page, list):
                page = page[0] if page else 0

            metadata = {
                "doc_id": doc_id,
                "type": chunk_type,
                "page": int(page),
                "section": str(chunk.get("section", "")),
                "chunk_idx": int(chunk_idx),
                "token_count": int(chunk.get("token_count", 0)),
            }

            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)

        if not ids:
            logger.warning(f"No valid chunks to add for doc_id={doc_id}")
            return 0

        # Upsert in batches of 500 to avoid memory issues
        batch_size = 500
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            total_added += len(batch_ids)

        logger.info(f"Added {total_added} chunks for doc_id={doc_id} to vector store")
        return total_added

    def add_from_bundle(self, bundle) -> int:
        """
        Add all chunks from an ingestion DocumentBundle.

        Args:
            bundle: DocumentBundle from ingestion pipeline

        Returns:
            Total number of embedded chunks
        """
        total = 0
        all_chunks = (
            bundle.text_chunks
            + bundle.image_elements
            + bundle.formula_elements
        )
        total += self.add_chunks(all_chunks, bundle.doc_id)
        return total

    # ── Search / Retrieval ───────────────────────

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        doc_id: Optional[str] = None,
        chunk_types: Optional[list[str]] = None,
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Args:
            query: User query string
            top_k: Number of results to return
            doc_id: Filter to a specific document (None = all docs)
            chunk_types: Filter to specific types ["text", "image", "formula", "table"]
            score_threshold: Minimum similarity score (0–1 for cosine)

        Returns:
            List of result dicts with keys:
              text, score, doc_id, page, section, type, chunk_idx
        """
        collection = self._get_collection()

        # Build metadata filter
        where: dict = {}
        if doc_id and chunk_types:
            where = {
                "$and": [
                    {"doc_id": {"$eq": doc_id}},
                    {"type": {"$in": chunk_types}},
                ]
            }
        elif doc_id:
            where = {"doc_id": {"$eq": doc_id}}
        elif chunk_types:
            where = {"type": {"$in": chunk_types}}

        query_params: dict = {
            "query_texts": [query],
            "n_results": min(top_k, max(collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        try:
            results = collection.query(**query_params)
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

        output: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0=identical, 2=opposite → convert to similarity
            similarity = 1.0 - (dist / 2.0)
            if similarity < score_threshold:
                continue
            output.append({
                "text": doc,
                "score": round(float(similarity), 4),
                "doc_id": meta.get("doc_id", ""),
                "page": meta.get("page", 0),
                "section": meta.get("section", ""),
                "type": meta.get("type", "text"),
                "chunk_idx": meta.get("chunk_idx", 0),
            })

        # Sort by score descending
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    # ── Management ────────────────────────────────

    def list_documents(self) -> list[str]:
        """Return a list of unique doc_ids in the collection."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["metadatas"])
        doc_ids = list({m.get("doc_id", "") for m in result["metadatas"] if m})
        return sorted(doc_ids)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a specific doc_id."""
        collection = self._get_collection()
        collection.delete(where={"doc_id": {"$eq": doc_id}})
        logger.info(f"Deleted all chunks for doc_id={doc_id}")

    def reset(self) -> None:
        """Delete and recreate the collection (full wipe)."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Collection '{self.collection_name}' has been reset")

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._get_collection().count()

    def is_document_indexed(self, doc_id: str) -> bool:
        """Check if a document has already been indexed."""
        return doc_id in self.list_documents()


# ─────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────

_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the module-level singleton VectorStore."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
