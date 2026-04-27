"""
multimodal/formula_handler.py
==============================
Mathematical formula detection and LaTeX extraction.

Pipeline:
  1. Scan text elements for math-like patterns using regex heuristics
  2. For matched regions, attempt LaTeX OCR using pix2tex (local) or MathPix (cloud)
  3. Return formula elements with LaTeX representation + surrounding context

pix2tex reference: https://github.com/lukas-blecher/LaTeX-OCR
MathPix API: https://mathpix.com/docs/api
"""

import logging
import re
import io
from typing import Optional
from pathlib import Path

from PIL import Image

from config.settings import (
    USE_LOCAL_FORMULA_OCR,
    MATHPIX_APP_ID,
    MATHPIX_APP_KEY,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MATH PATTERN DETECTION
# ─────────────────────────────────────────────

# Patterns that strongly suggest mathematical content
MATH_PATTERNS = [
    r"\$[^$]+\$",                          # LaTeX inline math: $...$
    r"\$\$[^$]+\$\$",                      # LaTeX display math: $$...$$
    r"\\begin\{equation\}.*?\\end\{equation\}",  # LaTeX equation environment
    r"\\begin\{align\}.*?\\end\{align\}",
    r"\b[A-Za-z]\s*[=<>≤≥≠]\s*[A-Za-z0-9\+\-\*\/\^\(\)]+",  # algebraic
    r"\b(?:sum|prod|int|lim|log|exp|sin|cos|tan|det|max|min)"
    r"\s*(?:[_^]?\{[^}]+\})*",             # math functions
    r"[∑∏∫∂∇∞α-ωΑ-Ω±×÷≈≡∈∉⊂⊃∪∩]",       # unicode math symbols
    r"\d+\s*[\+\-\*\/\^]\s*\d+",           # numeric expressions
    r"[A-Za-z_]\w*\s*\([^)]*\)\s*=",       # function definitions
]

_MATH_REGEX = re.compile("|".join(MATH_PATTERNS), re.DOTALL | re.IGNORECASE)

# Minimum character length to consider a formula worthy of extraction
MIN_FORMULA_LENGTH = 5


def detect_math_in_text(text: str) -> list[tuple[int, int, str]]:
    """
    Detect mathematical expressions in plain text.

    Returns:
        List of (start, end, matched_text) tuples
    """
    matches = []
    for m in _MATH_REGEX.finditer(text):
        content = m.group().strip()
        if len(content) >= MIN_FORMULA_LENGTH:
            matches.append((m.start(), m.end(), content))
    return matches


def has_math(text: str) -> bool:
    """Quick test: does this text contain mathematical content?"""
    return bool(_MATH_REGEX.search(text))


# ─────────────────────────────────────────────
# LaTeX OCR (pix2tex — local)
# ─────────────────────────────────────────────

class LocalFormulaOCR:
    """
    Wrapper around pix2tex for offline LaTeX OCR from images.
    Falls back gracefully if pix2tex is not installed.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from pix2tex.cli import LatexOCR
            self._model = LatexOCR()
            logger.info("pix2tex LatexOCR loaded successfully")
        except ImportError:
            logger.warning(
                "pix2tex not installed. Formula image OCR disabled. "
                "Install with: pip install pix2tex"
            )
            self._model = None

    def ocr(self, image: Image.Image) -> str:
        """Extract LaTeX from a PIL image of a formula."""
        self._load()
        if self._model is None:
            return ""
        try:
            result = self._model(image)
            return result or ""
        except Exception as exc:
            logger.warning(f"pix2tex OCR failed: {exc}")
            return ""


# ─────────────────────────────────────────────
# LaTeX OCR (MathPix — cloud fallback)
# ─────────────────────────────────────────────

class MathPixOCR:
    """Cloud-based formula OCR via MathPix API."""

    API_URL = "https://api.mathpix.com/v3/text"

    def __init__(self, app_id: str = MATHPIX_APP_ID, app_key: str = MATHPIX_APP_KEY):
        self.app_id = app_id
        self.app_key = app_key

    def ocr(self, image: Image.Image) -> str:
        """Send image to MathPix and return LaTeX."""
        if not (self.app_id and self.app_key):
            logger.debug("MathPix credentials not set; skipping cloud OCR")
            return ""
        try:
            import requests
            import base64

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post(
                self.API_URL,
                json={
                    "src": f"data:image/png;base64,{img_b64}",
                    "formats": ["latex_simplified"],
                },
                headers={
                    "app_id": self.app_id,
                    "app_key": self.app_key,
                    "Content-type": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("latex_simplified", "")
        except Exception as exc:
            logger.warning(f"MathPix OCR failed: {exc}")
            return ""


# ─────────────────────────────────────────────
# FORMULA HANDLER
# ─────────────────────────────────────────────

class FormulaHandler:
    """
    Detects and extracts mathematical formulas from PDF text elements.

    For text-based formulas: regex detection + plain-text representation.
    For image-based formulas: pix2tex / MathPix OCR for LaTeX.

    Usage:
        handler = FormulaHandler()
        clean_elements, formula_elements = handler.extract_from_text_elements(elements)
    """

    def __init__(self):
        self._local_ocr = LocalFormulaOCR() if USE_LOCAL_FORMULA_OCR else None
        self._mathpix = MathPixOCR()

    def extract_from_text_elements(
        self, elements: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Scan text elements, extract formula segments into separate formula elements.

        Args:
            elements: List of text element dicts from PDFParser

        Returns:
            (cleaned_text_elements, formula_elements)
        """
        clean_text = []
        formulas = []
        formula_idx = 0

        for elem in elements:
            if elem["type"] != "text":
                clean_text.append(elem)
                continue

            text = elem["content"]
            matches = detect_math_in_text(text)

            if not matches:
                clean_text.append(elem)
                continue

            # For each detected formula, create a formula element
            for start, end, formula_text in matches:
                formula_elem = self._make_formula_element(
                    formula_text=formula_text,
                    context=text,          # full surrounding text as context
                    page=elem["page"],
                    section=elem.get("section", ""),
                    formula_idx=formula_idx,
                )
                formulas.append(formula_elem)
                formula_idx += 1

            # Keep the text element too (formulas are extracted additively)
            clean_text.append(elem)

        return clean_text, formulas

    def ocr_formula_image(self, image: Image.Image) -> str:
        """
        Extract LaTeX from an image containing a formula.
        Tries local pix2tex first; falls back to MathPix.
        """
        latex = ""
        if self._local_ocr:
            latex = self._local_ocr.ocr(image)
        if not latex and self._mathpix:
            latex = self._mathpix.ocr(image)
        return latex or "[LaTeX OCR unavailable]"

    def _make_formula_element(
        self,
        formula_text: str,
        context: str,
        page: int,
        section: str,
        formula_idx: int,
    ) -> dict:
        """
        Build a formula element dict.

        Attempts basic cleanup of common LaTeX patterns.
        """
        # Try to clean up raw text formula to rudimentary LaTeX
        latex = self._text_to_latex(formula_text)
        return {
            "type": "formula",
            "content": f"Formula: {formula_text}",
            "latex": latex,
            "raw_text": formula_text,
            "context": context[:400],  # first 400 chars of surrounding text
            "page": page,
            "section": section,
            "formula_idx": formula_idx,
        }

    @staticmethod
    def _text_to_latex(text: str) -> str:
        """
        Rudimentary conversion of detected text formulas to LaTeX-ish format.
        A full conversion requires proper OCR; this is a best-effort heuristic.
        """
        # Already wrapped in LaTeX delimiters?
        if text.startswith("$") or text.startswith("\\"):
            return text

        # Replace common unicode math with LaTeX equivalents
        replacements = {
            "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
            "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty",
            "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
            "ε": r"\epsilon", "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu",
            "π": r"\pi", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi",
            "ω": r"\omega", "≈": r"\approx", "≡": r"\equiv", "≤": r"\leq",
            "≥": r"\geq", "≠": r"\neq", "∈": r"\in", "∉": r"\notin",
            "±": r"\pm", "×": r"\times", "÷": r"\div",
        }
        for char, latex in replacements.items():
            text = text.replace(char, latex)

        return f"${text}$"
