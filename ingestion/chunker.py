"""
ingestion/chunker.py
=====================
Token-aware text chunking with overlap.

Chunking strategy:
  - Split text into chunks of at most CHUNK_SIZE tokens
  - Use CHUNK_OVERLAP tokens of overlap between consecutive chunks
  - Respect natural sentence and paragraph boundaries where possible
  - Preserve metadata (page, section, source type) on each chunk

Tokenization uses the tiktoken (OpenAI-compatible) tokenizer for
reliable token counting regardless of the downstream LLM.
"""

import logging
import re
from typing import Any

import tiktoken

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH

logger = logging.getLogger(__name__)

# Use cl100k_base (GPT-4 / text-embedding-ada-002 tokenizer) as a universal reference
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────
# LOW-LEVEL TOKEN UTILITIES
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def token_slice(text: str, start: int, end: int) -> str:
    """Return the substring corresponding to tokens[start:end]."""
    tokens = _TOKENIZER.encode(text, disallowed_special=())
    return _TOKENIZER.decode(tokens[start:end])


# ─────────────────────────────────────────────
# TEXT SPLITTER
# ─────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter using regex.
    Handles common academic abbreviations (e.g., et al., Fig., Eq.).
    """
    # Protect common abbreviations from being split
    abbreviations = r"(et al|Fig|Eq|Dr|Prof|vs|i\.e|e\.g|cf|approx|est)\."
    text = re.sub(abbreviations, lambda m: m.group().replace(".", "<!DOT!>"), text)

    # Split on sentence-ending punctuation followed by whitespace + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # Restore abbreviations
    sentences = [s.replace("<!DOT!>", ".") for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


class TokenAwareChunker:
    """
    Splits a list of text blocks into overlapping token-aware chunks.

    Each output chunk contains:
      - text: the chunk content
      - page: source page number(s)
      - section: detected section heading
      - type: "text" | "table" | "formula"
      - chunk_idx: sequential index
      - token_count: number of tokens in this chunk
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_length: int = MIN_CHUNK_LENGTH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk_elements(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a list of parsed PDF elements and return chunked text blocks.

        Tables and formulas are kept as atomic chunks (not split further).
        Text blocks are sentence-split and assembled into token-size windows.
        """
        chunks: list[dict] = []

        # Separate text from atomic elements
        text_elements = [e for e in elements if e["type"] == "text"]
        atomic_elements = [e for e in elements if e["type"] in ("table", "formula")]

        # ── Chunk text elements ──
        text_chunks = self._chunk_text_elements(text_elements)
        chunks.extend(text_chunks)

        # ── Pass atomic elements through as-is ──
        for idx, elem in enumerate(atomic_elements):
            content = elem["content"]
            if count_tokens(content) < self.min_chunk_length:
                continue
            chunks.append({
                "text": content,
                "page": elem["page"],
                "section": elem.get("section", ""),
                "type": elem["type"],
                "chunk_idx": len(chunks),
                "token_count": count_tokens(content),
                "metadata": elem.get("metadata", {}),
            })

        # Re-index
        for i, chunk in enumerate(chunks):
            chunk["chunk_idx"] = i

        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks

    def _chunk_text_elements(self, elements: list[dict]) -> list[dict]:
        """
        Join all text elements into one stream, then apply sliding window chunking.
        Tracks page boundaries to assign correct page numbers to each chunk.
        """
        # Build a flat list of (sentence, page, section)
        sentence_meta: list[tuple[str, int, str]] = []
        for elem in elements:
            sentences = _split_into_sentences(elem["content"])
            for sent in sentences:
                if len(sent.strip()) >= self.min_chunk_length:
                    sentence_meta.append((sent, elem["page"], elem.get("section", "")))

        if not sentence_meta:
            return []

        chunks: list[dict] = []
        buffer_sentences: list[tuple[str, int, str]] = []
        buffer_tokens = 0

        for sent, page, section in sentence_meta:
            sent_tokens = count_tokens(sent)

            # If adding this sentence would exceed chunk_size, flush the buffer
            if buffer_tokens + sent_tokens > self.chunk_size and buffer_sentences:
                chunk = self._flush_buffer(buffer_sentences)
                chunks.append(chunk)

                # Keep overlap: retain trailing sentences that fit in overlap budget
                overlap_tokens = 0
                new_buffer: list[tuple[str, int, str]] = []
                for s, p, sec in reversed(buffer_sentences):
                    t = count_tokens(s)
                    if overlap_tokens + t <= self.chunk_overlap:
                        new_buffer.insert(0, (s, p, sec))
                        overlap_tokens += t
                    else:
                        break
                buffer_sentences = new_buffer
                buffer_tokens = overlap_tokens

            buffer_sentences.append((sent, page, section))
            buffer_tokens += sent_tokens

        # Flush the last buffer
        if buffer_sentences:
            chunks.append(self._flush_buffer(buffer_sentences))

        return chunks

    def _flush_buffer(self, buffer: list[tuple[str, int, str]]) -> dict:
        """Convert a sentence buffer into a chunk dict."""
        text = " ".join(s for s, _, _ in buffer)
        pages = sorted(set(p for _, p, _ in buffer))
        # Use the most recent section
        sections = [sec for _, _, sec in buffer if sec]
        section = sections[-1] if sections else ""
        return {
            "text": text.strip(),
            "page": pages[0] if len(pages) == 1 else pages,
            "section": section,
            "type": "text",
            "chunk_idx": 0,                    # will be re-indexed later
            "token_count": count_tokens(text),
        }


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def chunk_elements(
    elements: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Top-level helper to chunk parsed PDF elements."""
    chunker = TokenAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_elements(elements)
