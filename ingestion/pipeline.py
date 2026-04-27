"""
ingestion/pipeline.py
======================
Orchestrates the full document ingestion pipeline:

  1. PDF parsing  → raw elements (text, images, tables, links)
  2. Image processing → captions added to image elements
  3. Formula detection → LaTeX extracted from text elements
  4. Chunking → text broken into token-aware chunks
  5. Building a unified document representation

The output is a DocumentBundle, a structured object that can be
passed to the embeddings module or stored to disk.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import chunk_elements
from multimodal.image_captioner import ImageCaptioner
from multimodal.formula_handler import FormulaHandler
from multimodal.image_extractor import save_all_images
from config.settings import UPLOAD_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class DocumentBundle:
    """
    Unified document representation produced by the ingestion pipeline.

    Attributes:
        doc_id:     Unique document identifier (typically filename stem)
        filename:   Original PDF filename
        metadata:   PDF metadata (author, title, page count, etc.)
        text_chunks: List of chunked text dicts (ready for embedding)
        image_elements: List of image dicts with captions
        formula_elements: List of formula dicts with LaTeX
        table_elements: List of table dicts in markdown format
        links: List of hyperlinks
        ingestion_time_sec: Duration of ingestion in seconds
    """
    doc_id: str
    filename: str
    metadata: dict = field(default_factory=dict)
    text_chunks: list[dict] = field(default_factory=list)
    image_elements: list[dict] = field(default_factory=list)
    formula_elements: list[dict] = field(default_factory=list)
    table_elements: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    ingestion_time_sec: float = 0.0

    @property
    def total_chunks(self) -> int:
        return (
            len(self.text_chunks)
            + len(self.image_elements)
            + len(self.formula_elements)
            + len(self.table_elements)
        )

    def summary(self) -> str:
        return (
            f"Document: {self.filename}\n"
            f"  Pages:          {self.metadata.get('page_count', '?')}\n"
            f"  Text chunks:    {len(self.text_chunks)}\n"
            f"  Images:         {len(self.image_elements)}\n"
            f"  Formulas:       {len(self.formula_elements)}\n"
            f"  Tables:         {len(self.table_elements)}\n"
            f"  Links:          {len(self.links)}\n"
            f"  Total chunks:   {self.total_chunks}\n"
            f"  Ingestion time: {self.ingestion_time_sec:.1f}s"
        )


# ─────────────────────────────────────────────
# INGESTION PIPELINE
# ─────────────────────────────────────────────

class IngestionPipeline:
    """
    Full document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()
        bundle = pipeline.run("paper.pdf")
        print(bundle.summary())
    """

    def __init__(
        self,
        enable_image_captioning: bool = True,
        enable_formula_detection: bool = True,
        min_image_size: int = 100,
    ):
        self.enable_image_captioning = enable_image_captioning
        self.enable_formula_detection = enable_formula_detection
        self.min_image_size = min_image_size

        # Lazy-load heavy models
        self._captioner: ImageCaptioner | None = None
        self._formula_handler: FormulaHandler | None = None

    # ── Lazy model loaders ───────────────────────

    def _get_captioner(self) -> ImageCaptioner:
        if self._captioner is None:
            logger.info("Loading image captioning model...")
            self._captioner = ImageCaptioner()
        return self._captioner

    def _get_formula_handler(self) -> FormulaHandler:
        if self._formula_handler is None:
            logger.info("Loading formula handler...")
            self._formula_handler = FormulaHandler()
        return self._formula_handler

    # ── Main entry point ─────────────────────────

    def run(self, pdf_path: str | Path) -> DocumentBundle:
        """
        Run the full ingestion pipeline on a PDF file.

        Returns:
            DocumentBundle: structured document representation
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        logger.info(f"Starting ingestion pipeline for: {pdf_path.name}")

        # ── Step 1: Parse PDF ──
        parser = PDFParser(pdf_path, min_image_size=self.min_image_size)
        elements = parser.parse()
        metadata = parser.get_metadata()

        # ── Step 2: Separate element types ──
        raw_text_elements = [e for e in elements if e["type"] == "text"]
        image_elements = [e for e in elements if e["type"] == "image"]
        table_elements = [e for e in elements if e["type"] == "table"]
        link_elements = [e for e in elements if e["type"] == "link"]

        # ── Step 3: Formula detection from text ──
        formula_elements: list[dict] = []
        if self.enable_formula_detection and raw_text_elements:
            formula_handler = self._get_formula_handler()
            raw_text_elements, formula_elements = formula_handler.extract_from_text_elements(
                raw_text_elements
            )
            logger.info(f"Detected {len(formula_elements)} formula(s)")

        # ── Step 4: Chunk text elements ──
        text_chunks = chunk_elements(raw_text_elements + table_elements)

        doc_id = pdf_path.stem

        # ── Step 5: Save images to disk ──
        images_dir = UPLOAD_DIR / doc_id
        image_elements = save_all_images(image_elements, images_dir, doc_id)

        # ── Step 6: Image captioning ──
        if self.enable_image_captioning and image_elements:
            captioner = self._get_captioner()
            image_elements = captioner.caption_elements(image_elements)
            logger.info(f"Captioned {len(image_elements)} image(s)")

        ingestion_time = time.time() - start_time

        bundle = DocumentBundle(
            doc_id=doc_id,
            filename=pdf_path.name,
            metadata=metadata,
            text_chunks=text_chunks,
            image_elements=image_elements,
            formula_elements=formula_elements,
            table_elements=table_elements,
            links=link_elements,
            ingestion_time_sec=ingestion_time,
        )

        logger.info(f"Ingestion complete in {ingestion_time:.1f}s:\n{bundle.summary()}")
        return bundle


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str | Path,
    enable_image_captioning: bool = True,
    enable_formula_detection: bool = True,
) -> DocumentBundle:
    """Top-level convenience function for ingesting a single PDF."""
    pipeline = IngestionPipeline(
        enable_image_captioning=enable_image_captioning,
        enable_formula_detection=enable_formula_detection,
    )
    return pipeline.run(pdf_path)
