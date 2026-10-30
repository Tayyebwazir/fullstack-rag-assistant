import os 
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR /".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
COLLECTION_NAME = os.getenv("COLLECTION_NAME","PDFChunk")
WEAVIATE_MODE = os.getenv("WEAVIATE_MODE", os.getenv("WEAVIATE_MODEL", "local"))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
PDF_PATH =BASE_DIR/ os.getenv("PDF_PATH","data/your_doucment.pdf")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP","100"))
TOP_K = int(os.getenv("TOP_K","4"))