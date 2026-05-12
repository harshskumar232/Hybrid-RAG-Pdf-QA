import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR  = DATA_DIR / "chroma_db"
ANALYTICS_DB = DATA_DIR / "analytics.db"

for d in (UPLOADS_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 1024

# ── Embeddings ────────────────────────────────────────────────────────────────
# Multilingual model — handles German + English natively
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM   = 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 80

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE = 5    # ChromaDB results
TOP_K_BM25  = 5    # BM25 results
TOP_K_FINAL = 5    # after fusion

# ── Summarisation ─────────────────────────────────────────────────────────────
SUMMARY_MAX_CHUNKS = 20   # cap chunks fed to summariser

# ── Languages ─────────────────────────────────────────────────────────────────
SUPPORTED_LANGS = {"en": "English", "de": "German"}
DEFAULT_LANG    = "en"
