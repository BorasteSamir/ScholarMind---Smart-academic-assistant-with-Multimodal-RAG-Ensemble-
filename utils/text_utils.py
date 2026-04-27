"""
utils/text_utils.py
====================
Text processing utilities for cleaning and formatting strings.
"""

import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Clean raw extracted text:
      - Normalize unicode characters
      - Strip excessive whitespace
      - Normalize dashes and quotes
    """
    if not text:
        return ""

    # Normalize unicode (e.g. NFKC canonical decomposition)
    text = unicodedata.normalize("NFKC", text)

    # Replace en-dash / em-dash with hyphen
    text = re.sub(r"[\u2013\u2014]", "-", text)

    # Replace smart quotes with standard quotes
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)

    # Collapse multiple spaces or newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()

def truncate_text(text: str, max_chars: int = 150) -> str:
    """Truncate text to max_chars and add ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
