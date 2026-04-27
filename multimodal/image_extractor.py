"""
multimodal/image_extractor.py
==============================
Utilities for saving, loading, and managing extracted PDF images.

The PDFParser extracts raw image bytes; this module provides helpers to:
  - Convert raw bytes to PIL Images
  - Save images to disk with structured names
  - Load images back for display or re-processing
"""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
import io

logger = logging.getLogger(__name__)


def bytes_to_pil(image_bytes: bytes) -> Optional[Image.Image]:
    """Convert raw image bytes to a PIL Image object."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning(f"Failed to convert image bytes to PIL: {exc}")
        return None


def save_image(
    image_bytes: bytes,
    output_dir: Path,
    doc_id: str,
    page: int,
    img_idx: int,
    ext: str = "png",
) -> Optional[Path]:
    """
    Save an image extracted from a PDF to disk.

    Args:
        image_bytes: Raw image bytes from PDF extraction
        output_dir: Directory to save the image
        doc_id: Document identifier (used in filename)
        page: Page number the image came from
        img_idx: Image index on the page
        ext: Image file extension

    Returns:
        Path to saved image, or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{doc_id}_page{page:03d}_img{img_idx:02d}.{ext}"
    path = output_dir / filename
    try:
        img = bytes_to_pil(image_bytes)
        if img is None:
            return None
        img.save(path)
        logger.debug(f"Saved image: {path}")
        return path
    except Exception as exc:
        logger.error(f"Failed to save image {filename}: {exc}")
        return None


def load_image(path: Path | str) -> Optional[Image.Image]:
    """Load an image from disk as PIL Image."""
    try:
        return Image.open(str(path)).convert("RGB")
    except Exception as exc:
        logger.error(f"Failed to load image from {path}: {exc}")
        return None


def save_all_images(
    image_elements: list[dict],
    output_dir: Path,
    doc_id: str,
) -> list[dict]:
    """
    Save all image elements to disk and add 'file_path' key to each element.

    Args:
        image_elements: List of image element dicts from PDFParser
        output_dir: Directory to save images
        doc_id: Document identifier

    Returns:
        Updated image elements with 'file_path' added
    """
    updated = []
    for elem in image_elements:
        path = save_image(
            image_bytes=elem["content"],
            output_dir=output_dir,
            doc_id=doc_id,
            page=elem["page"],
            img_idx=elem["image_idx"],
            ext=elem.get("ext", "png"),
        )
        if path:
            elem = dict(elem)  # copy to avoid mutation
            elem["file_path"] = str(path)
        updated.append(elem)
    return updated
