"""
config/settings.py
==================
Central configuration for Smart Academic Assistant.
All values are driven by environment variables with sensible defaults.
Never hardcode secrets here — use .env file or OS environment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, CACHE_DIR, CHROMA_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LLM CONFIGURATION
# ─────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")          # "openai" | "local" | "anthropic"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")  # Ollama
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-fast")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ─────────────────────────────────────────────
# EMBEDDING CONFIGURATION
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cpu" | "cuda"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ─────────────────────────────────────────────
# CHUNKING CONFIGURATION
# ─────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))           # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))      # overlap tokens
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# ─────────────────────────────────────────────
# RETRIEVAL CONFIGURATION
# ─────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))

# ─────────────────────────────────────────────
# VECTOR STORE CONFIGURATION
# ─────────────────────────────────────────────
VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION", "academic_docs")

# ─────────────────────────────────────────────
# IMAGE CAPTIONING CONFIGURATION
# ─────────────────────────────────────────────
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
BLIP_DEVICE = os.getenv("BLIP_DEVICE", "cpu")
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "100"))   # minimum px (width or height)

# ─────────────────────────────────────────────
# FORMULA HANDLING
# ─────────────────────────────────────────────
# pix2tex for local LaTeX OCR; set MATHPIX_APP_ID/KEY for cloud fallback
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY", "")
USE_LOCAL_FORMULA_OCR = os.getenv("USE_LOCAL_FORMULA_OCR", "true").lower() == "true"

# ─────────────────────────────────────────────
# EVALUATION CONFIGURATION
# ─────────────────────────────────────────────
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.25"))
ROUGE_N = int(os.getenv("ROUGE_N", "2"))

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "assistant.log"

# ─────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────
APP_TITLE = "Smart Academic Assistant"
APP_VERSION = "1.0.0"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
