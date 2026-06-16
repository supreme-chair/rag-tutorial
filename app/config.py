"""Configuration for RAG system."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Chunking settings
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.01

# UI settings
UI_TITLE = "RAG Assistant - Russian Tech Blog Posts"
UI_ICON = "🤖"

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Multilingual model with Russian support