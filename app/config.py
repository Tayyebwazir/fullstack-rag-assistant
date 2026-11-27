import os 
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MultiTenantRAG")
WEAVIATE_MODE = os.getenv("WEAVIATE_MODE", os.getenv("WEAVIATE_MODEL", "cloud"))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "app/data/Attention is all you need.pdf")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.30"))
SEARCH_MODE = os.getenv("SEARCH_MODE", "vector").strip().lower()
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.50"))
HYBRID_SCORE_THRESHOLD = float(os.getenv("HYBRID_SCORE_THRESHOLD", "0.0"))



# CORS Configuration: load comma-separated trusted origins from .env with localhost defaults
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
env_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    if env_origins
    else DEFAULT_CORS_ORIGINS
)

# Security & Ingestion Safeguards
INGEST_RATE_LIMIT = int(os.getenv("INGEST_RATE_LIMIT", "5"))
INGEST_RATE_WINDOW_SECONDS = int(os.getenv("INGEST_RATE_WINDOW_SECONDS", "3600"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "500"))

# Security & Ingestion Safeguards
INGEST_RATE_LIMIT = int(os.getenv("INGEST_RATE_LIMIT", "5"))
INGEST_RATE_WINDOW_SECONDS = int(os.getenv("INGEST_RATE_WINDOW_SECONDS", "3600"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "500"))
