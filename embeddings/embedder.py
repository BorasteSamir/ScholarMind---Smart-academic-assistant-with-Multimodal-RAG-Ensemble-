"""
embeddings/embedder.py
=======================
Embedding generation using Sentence Transformers.

Converts text chunks (and image captions / formula text) into dense
vector representations for similarity search.

Model: sentence-transformers/all-MiniLM-L6-v2 (default) — 384 dimensions
       Swappable via EMBEDDING_MODEL env variable.

Design:
  - Batch encoding for efficiency
  - L2 / cosine normalization applied by default
  - Returns numpy arrays for FAISS compatibility
"""

import logging
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper around SentenceTransformer for text embedding.

    Supports:
      - Single text encoding
      - Batch encoding (more efficient)
      - Metadata-aware chunk encoding

    Usage:
        embedder = Embedder()
        vector = embedder.embed("Transformers are attention-based models")
        vectors = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None

    # ── Lazy loading ─────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"Embedding model loaded. Dimension: {self.embedding_dim}"
            )
        return self._model

    # ── Properties ───────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Return the embedding vector dimension."""
        model = self._load_model()
        try:
            return model.get_embedding_dimension()
        except AttributeError:
            return model.get_sentence_embedding_dimension()

    # ── Core embedding methods ────────────────────

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text
            normalize: L2-normalize the output vector (recommended for cosine sim)

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        model = self._load_model()
        with torch.no_grad():
            vector = model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        return np.array(vector, dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of input texts
            normalize: L2-normalize each output vector
            show_progress: Show tqdm progress bar

        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        model = self._load_model()
        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")

        with torch.no_grad():
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
            )
        return np.array(vectors, dtype=np.float32)

    # ── Chunk embedding ────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """
        Embed a list of chunk dicts (from the chunker / formula / image pipeline).

        Chooses the text field based on chunk type:
          - text/table: uses 'text' field
          - image:      uses 'caption' field
          - formula:    uses 'content' field (readable formula text)

        Returns:
            (embeddings, valid_chunks) — only chunks with non-empty text are returned
        """
        valid_chunks = []
        texts = []

        for chunk in chunks:
            chunk_type = chunk.get("type", "text")
            if chunk_type == "image":
                text = chunk.get("caption", "")
            elif chunk_type == "formula":
                content = chunk.get("content", "")
                latex = chunk.get("latex", "")
                text = f"{content} {latex}".strip()
            else:
                text = chunk.get("text", chunk.get("content", ""))

            if text.strip():
                texts.append(text)
                valid_chunks.append(chunk)

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        embeddings = self.embed_batch(texts)
        return embeddings, valid_chunks


# ─────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────

_default_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Return the module-level singleton Embedder instance."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


def embed_text(text: str) -> np.ndarray:
    """Embed a single text using the default embedder."""
    return get_embedder().embed(text)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a user query.
    Prepends 'query: ' prefix — some models improve with query-specific prefix.
    """
    prefixed = f"query: {query}"
    return get_embedder().embed(prefixed)
