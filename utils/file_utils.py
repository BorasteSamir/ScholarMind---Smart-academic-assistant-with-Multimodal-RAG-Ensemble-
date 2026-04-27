"""
utils/file_utils.py
====================
File handling utilities, especially for managing Streamlit uploads.
"""

import logging
from pathlib import Path

from config.settings import UPLOAD_DIR, MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file) -> Path | None:
    """
    Save a Streamlit UploadedFile to the local UPLOAD_DIR.
    Checks file size against MAX_UPLOAD_SIZE_MB.
    
    Returns:
        Path to the saved file, or None if failed/too large.
    """
    if uploaded_file is None:
        return None

    # Check size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        logger.error(
            f"Uploaded file '{uploaded_file.name}' ({file_size_mb:.1f} MB) "
            f"exceeds limit of {MAX_UPLOAD_SIZE_MB} MB."
        )
        return None

    # Save to disk
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / uploaded_file.name

    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {save_path}")
        return save_path
    except Exception as exc:
        logger.error(f"Failed to save uploaded file: {exc}")
        return None

def clear_upload_dir() -> None:
    """Delete all files in the UPLOAD_DIR."""
    count = 0
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
                count += 1
            except Exception as exc:
                logger.warning(f"Failed to delete {file_path}: {exc}")
    if count > 0:
        logger.info(f"Cleared {count} files from upload directory.")
