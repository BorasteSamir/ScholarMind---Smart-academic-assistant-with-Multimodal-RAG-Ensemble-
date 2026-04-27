"""
evaluation/logger.py
=====================
Centralized logging setup for the Smart Academic Assistant.

Features:
  - Formatted console output (colorized via formatter if desired)
  - File rotation for persistence
  - Configurable log levels via environment variables
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import LOG_LEVEL, LOG_FILE

# ─────────────────────────────────────────────
# LOG FORMATTING
# ─────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Simple colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[94m",     # Blue
        "INFO": "\033[92m",      # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "CRITICAL": "\033[1;91m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ─────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────

def setup_logging(
    log_level: str = LOG_LEVEL,
    log_file: Path | str = LOG_FILE,
) -> None:
    """
    Initialize global logging configuration.
    Call this once at application startup.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()

    # ── Console Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ── File Handler (Rotating) ──
    # 5 MB per file, max 3 files
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.info(f"Logging initialized (Level: {log_level.upper()})")


# ─────────────────────────────────────────────
# AUTO-INITIALIZE ON IMPORT
# ─────────────────────────────────────────────

# Automatically setup logging when this module is imported.
# It's safe to call setup_logging() multiple times if needed.
setup_logging()
