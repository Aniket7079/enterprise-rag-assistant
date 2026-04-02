import os
from pathlib import Path

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_internal_docs")
TOP_K = int(os.getenv("TOP_K", "4"))
DEFAULT_THREAD_ID = "demo-thread"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
