"""
embeddings/cache.py
====================
Embedding cache using diskcache for persistent on-disk caching.

Avoids re-computing embeddings for chunks already processed.
Cache key = SHA-256 hash of (model_name + text content).

Benefits:
  - Skip expensive embedding calls on repeat ingestions or app restarts
  - Transparent: cached embeddings are indistinguishable from fresh ones
  - Thread-safe due to diskcache's locking mechanism
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import diskcache as dc

from config.settings import CACHE_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Persistent disk-based embedding cache.

    Usage:
        cache = EmbeddingCache()
        vec = cache.get("some text")   # None if not cached
        cache.set("some text", embedding_array)
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        model_name: str = EMBEDDING_MODEL,
        size_limit_gb: float = 5.0,
    ):
        cache_path = Path(cache_dir) / "embedding_cache"
        self._cache = dc.Cache(
            str(cache_path),
            size_limit=int(size_limit_gb * 1024 ** 3),
        )
        self.model_name = model_name
        logger.info(f"Embedding cache initialized at: {cache_path}")

    def _make_key(self, text: str) -> str:
        """Generate a deterministic cache key from model name + text."""
        content = f"{self.model_name}::{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding.

        Returns:
            numpy array if cached, None otherwise
        """
        key = self._make_key(text)
        result = self._cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for text: {text[:40]}...")
        return result

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache."""
        key = self._make_key(text)
        self._cache.set(key, embedding)

    def get_or_compute(
        self,
        text: str,
        compute_fn,
    ) -> np.ndarray:
        """
        Get from cache or compute using the provided function.

        Args:
            text: The text to embed
            compute_fn: Callable that takes text and returns np.ndarray

        Returns:
            np.ndarray embedding
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding

    def batch_get_or_compute(
        self,
        texts: list[str],
        batch_compute_fn,
    ) -> np.ndarray:
        """
        Batch version: returns cached embeddings where available,
        computes missing ones in a single batch call.

        Args:
            texts: List of texts
            batch_compute_fn: Callable that takes list[str] → np.ndarray

        Returns:
            2D np.ndarray of shape (len(texts), embedding_dim)
        """
        results: list[Optional[np.ndarray]] = [self.get(t) for t in texts]
        missing_indices = [i for i, r in enumerate(results) if r is None]
        missing_texts = [texts[i] for i in missing_indices]

        if missing_texts:
            logger.info(
                f"Cache miss for {len(missing_texts)}/{len(texts)} texts, computing..."
            )
            new_embeddings = batch_compute_fn(missing_texts)
            for idx, text, emb in zip(missing_indices, missing_texts, new_embeddings):
                self.set(text, emb)
                results[idx] = emb

        return np.stack(results, axis=0)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size_bytes": self._cache.volume(),
            "num_entries": len(self._cache),
            "model": self.model_name,
        }

    def close(self) -> None:
        """Close the cache (release file locks)."""
        self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────

_default_cache: Optional[EmbeddingCache] = None


def get_cache() -> EmbeddingCache:
    """Return the module-level singleton EmbeddingCache (opened once)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbeddingCache()
    return _default_cache
