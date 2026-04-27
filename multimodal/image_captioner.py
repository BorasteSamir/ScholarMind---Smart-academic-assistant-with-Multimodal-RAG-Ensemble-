"""
multimodal/image_captioner.py
==============================
Image captioning using Salesforce BLIP (Bootstrapping Language-Image Pre-training).

BLIP generates natural-language captions from images extracted from PDFs.
Captions are then embedded alongside text chunks to enable multimodal retrieval.

Model: Salesforce/blip-image-captioning-base (or large variant)
Device: configurable (cpu / cuda)

Reference:
  Li et al. (2022). BLIP: Bootstrapping Language-Image Pre-training for
  Unified Vision-Language Understanding and Generation.
"""

import logging
from typing import Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from config.settings import BLIP_MODEL, BLIP_DEVICE
from multimodal.image_extractor import bytes_to_pil

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    Generates captions for images using BLIP.

    Usage:
        captioner = ImageCaptioner()
        caption = captioner.caption(pil_image)
    """

    def __init__(
        self,
        model_name: str = BLIP_MODEL,
        device: str = BLIP_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self._processor: Optional[BlipProcessor] = None
        self._model: Optional[BlipForConditionalGeneration] = None

    # ── Lazy loading ─────────────────────────────

    def _load_model(self):
        """Load BLIP model and processor on first use."""
        if self._model is not None:
            return
        logger.info(f"Loading BLIP model: {self.model_name} on {self.device}")
        try:
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()
            logger.info("BLIP model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load BLIP model: {exc}")
            raise

    # ── Core captioning ──────────────────────────

    def caption(self, image: Image.Image, prompt: str = "") -> str:
        """
        Generate a caption for a PIL Image.

        Args:
            image: PIL Image to caption
            prompt: Optional conditional prompt (e.g., "a diagram of")

        Returns:
            Generated caption string
        """
        self._load_model()
        try:
            if prompt:
                inputs = self._processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self._processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=150)

            caption = self._processor.decode(output_ids[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as exc:
            logger.warning(f"Caption generation failed: {exc}")
            return "Image from academic document"

    def caption_bytes(self, image_bytes: bytes) -> str:
        """Caption an image given as raw bytes."""
        pil_image = bytes_to_pil(image_bytes)
        if pil_image is None:
            return "Image from academic document (could not decode)"
        return self.caption(pil_image)

    # ── Batch processing ─────────────────────────

    def caption_elements(self, image_elements: list[dict]) -> list[dict]:
        """
        Add captions to a list of image elements in-place (returns updated copies).

        Each element must have 'content' (bytes) and 'page' (int) keys.
        The 'caption' key is added/updated on each element.

        Args:
            image_elements: List of image dicts from PDFParser

        Returns:
            Updated image element list with captions filled in
        """
        updated = []
        for elem in image_elements:
            elem = dict(elem)  # copy to avoid mutation
            try:
                caption = self.caption_bytes(elem["content"])
                elem["caption"] = caption
                logger.debug(f"Page {elem['page']} image captioned: {caption[:60]}...")
            except Exception as exc:
                logger.warning(f"Could not caption image on page {elem.get('page')}: {exc}")
                elem["caption"] = "Academic document image"
            updated.append(elem)
        return updated


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

_default_captioner: Optional[ImageCaptioner] = None


def get_default_captioner() -> ImageCaptioner:
    """Return a module-level singleton ImageCaptioner."""
    global _default_captioner
    if _default_captioner is None:
        _default_captioner = ImageCaptioner()
    return _default_captioner


def caption_image(image: Image.Image) -> str:
    """Caption a single PIL Image using the default captioner."""
    return get_default_captioner().caption(image)
