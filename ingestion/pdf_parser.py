"""
ingestion/pdf_parser.py
========================
PDF parsing module using PyMuPDF (fitz).

Extracts:
  - Structured text (with page/block metadata)
  - Images (with bounding box info)
  - Tables (via pdfplumber for accurate tabular data)
  - Hyperlinks

Each extracted element is returned as a typed dict so downstream
modules can route the data correctly (text → chunker, images → multimodal, etc.)
"""

import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TYPED OUTPUT STRUCTURES
# ─────────────────────────────────────────────

def _make_text_block(
    text: str,
    page: int,
    block_idx: int,
    bbox: tuple,
    section: str = "",
) -> dict[str, Any]:
    return {
        "type": "text",
        "content": text.strip(),
        "page": page,
        "block_idx": block_idx,
        "bbox": bbox,
        "section": section,
    }


def _make_image_block(
    image_bytes: bytes,
    page: int,
    image_idx: int,
    xref: int,
    width: int,
    height: int,
    ext: str,
) -> dict[str, Any]:
    return {
        "type": "image",
        "content": image_bytes,          # raw bytes
        "page": page,
        "image_idx": image_idx,
        "xref": xref,
        "width": width,
        "height": height,
        "ext": ext,
        "caption": "",                   # filled by image_captioner
    }


def _make_table_block(
    rows: list[list[str]],
    page: int,
    table_idx: int,
    bbox: tuple,
) -> dict[str, Any]:
    """Represent a table as markdown-formatted text for embedding."""
    # Convert to markdown
    if not rows:
        return {}
    headers = rows[0]
    body = rows[1:]
    header_line = " | ".join(str(h or "") for h in headers)
    separator = " | ".join(["---"] * len(headers))
    body_lines = [" | ".join(str(c or "") for c in row) for row in body]
    markdown_table = "\n".join([header_line, separator] + body_lines)
    return {
        "type": "table",
        "content": markdown_table,
        "raw_rows": rows,
        "page": page,
        "table_idx": table_idx,
        "bbox": bbox,
    }


def _make_link_block(uri: str, page: int) -> dict[str, Any]:
    return {
        "type": "link",
        "content": uri,
        "page": page,
    }


# ─────────────────────────────────────────────
# SECTION DETECTOR (heuristic)
# ─────────────────────────────────────────────

def _detect_section(block: dict) -> str:
    """
    Heuristic: large bold-like short lines are likely section headers.
    Returns the detected section name or empty string.
    """
    text = block.get("text", "").strip()
    spans = block.get("lines", [{}])[0].get("spans", [{}]) if block.get("lines") else []
    if not text:
        return ""
    # Short, potentially uppercase text → likely a heading
    if len(text) < 120 and (text.isupper() or (spans and spans[0].get("flags", 0) & 16)):
        return text
    return ""


# ─────────────────────────────────────────────
# MAIN PARSER CLASS
# ─────────────────────────────────────────────

class PDFParser:
    """
    Parses a PDF file and returns structured elements.

    Usage:
        parser = PDFParser("paper.pdf")
        elements = parser.parse()
        # elements: list of dicts with type in {text, image, table, link}
    """

    def __init__(self, pdf_path: str | Path, min_image_size: int = 100):
        self.pdf_path = Path(pdf_path)
        self.min_image_size = min_image_size
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ── Public API ───────────────────────────────

    def parse(self) -> list[dict[str, Any]]:
        """
        Full parse: text + images + tables + links.
        Returns a list of typed element dicts.
        """
        logger.info(f"Parsing PDF: {self.pdf_path.name}")
        elements: list[dict] = []

        elements.extend(self._extract_text_and_images())
        elements.extend(self._extract_tables())
        elements.extend(self._extract_links())

        logger.info(
            f"Parsed {self.pdf_path.name}: "
            f"{sum(1 for e in elements if e['type']=='text')} text blocks, "
            f"{sum(1 for e in elements if e['type']=='image')} images, "
            f"{sum(1 for e in elements if e['type']=='table')} tables, "
            f"{sum(1 for e in elements if e['type']=='link')} links"
        )
        return elements

    def get_page_count(self) -> int:
        with fitz.open(str(self.pdf_path)) as doc:
            return doc.page_count

    def get_metadata(self) -> dict:
        with fitz.open(str(self.pdf_path)) as doc:
            meta = doc.metadata or {}
        meta["page_count"] = self.get_page_count()
        meta["filename"] = self.pdf_path.name
        return meta

    # ── Private Helpers ──────────────────────────

    def _extract_text_and_images(self) -> list[dict]:
        """Use PyMuPDF to extract text blocks and embedded images."""
        elements: list[dict] = []
        current_section = ""

        with fitz.open(str(self.pdf_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                # ── Text blocks ──
                blocks = page.get_text("dict")["blocks"]
                for block_idx, block in enumerate(blocks):
                    if block.get("type") == 0:  # type 0 = text
                        text = " ".join(
                            span["text"]
                            for line in block.get("lines", [])
                            for span in line.get("spans", [])
                        ).strip()

                        if not text:
                            continue

                        # Detect section headers
                        section_candidate = _detect_section(block)
                        if section_candidate:
                            current_section = section_candidate

                        elem = _make_text_block(
                            text=text,
                            page=page_num,
                            block_idx=block_idx,
                            bbox=tuple(block["bbox"]),
                            section=current_section,
                        )
                        elements.append(elem)

                # ── Images ──
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        width = base_image["width"]
                        height = base_image["height"]
                        # Skip tiny images (icons, decorations)
                        if width < self.min_image_size or height < self.min_image_size:
                            continue
                        elem = _make_image_block(
                            image_bytes=base_image["image"],
                            page=page_num,
                            image_idx=img_idx,
                            xref=xref,
                            width=width,
                            height=height,
                            ext=base_image["ext"],
                        )
                        elements.append(elem)
                    except Exception as exc:
                        logger.warning(f"Could not extract image xref={xref}: {exc}")

        return elements

    def _extract_tables(self) -> list[dict]:
        """Use pdfplumber for accurate table extraction."""
        elements: list[dict] = []
        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        bbox = (0, 0, page.width, page.height)  # approximate
                        elem = _make_table_block(
                            rows=table,
                            page=page_num,
                            table_idx=table_idx,
                            bbox=bbox,
                        )
                        if elem:
                            elements.append(elem)
        except Exception as exc:
            logger.error(f"Table extraction failed: {exc}")
        return elements

    def _extract_links(self) -> list[dict]:
        """Extract hyperlinks from PDF annotations."""
        elements: list[dict] = []
        try:
            with fitz.open(str(self.pdf_path)) as doc:
                for page_num, page in enumerate(doc, start=1):
                    links = page.get_links()
                    for link in links:
                        uri = link.get("uri", "")
                        if uri:
                            elements.append(_make_link_block(uri=uri, page=page_num))
        except Exception as exc:
            logger.error(f"Link extraction failed: {exc}")
        return elements


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path, min_image_size: int = 100) -> list[dict[str, Any]]:
    """Top-level helper to parse a PDF and return all elements."""
    parser = PDFParser(pdf_path, min_image_size=min_image_size)
    return parser.parse()
