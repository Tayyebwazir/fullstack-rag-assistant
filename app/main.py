"""FastAPI application entry point wiring CORS middleware and feature routers."""

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.dependencies import require_tenant_permission
from app.routers import chat, documents, users
from app.routers.documents import get_sources as _get_sources_endpoint
from app.routers.documents import ingest_document as _ingest_document_endpoint
from app.schemas import DocIdRequest
from app.services.auth import get_current_user
from app.services.vector_store import close_weaviate_client, get_weaviate_client
from app.services.gemini_client import get_gemini_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages shared client startup and graceful shutdown."""
    try:
        get_weaviate_client()
    except Exception as exc:
        print(f"[Lifespan] Weaviate startup note: {exc}")
    try:
        get_gemini_client()
    except Exception as exc:
        print(f"[Lifespan] Gemini client startup note: {exc}")
    yield
    close_weaviate_client()


app = FastAPI(
    title="Multi-Tenant RAG Assistant API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

app.include_router(users.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def home():
    return {"message": "Multi-Tenant RAG Assistant API is running"}


@app.get("/sources")
def get_sources(
    tenant_id: str | None = None,
    user: dict = Depends(get_current_user),
):
    """Secured root alias for /documents/sources."""
    return _get_sources_endpoint(tenant_id=tenant_id, user=user)


@app.post("/ingest")
def ingest_root(
    req: DocIdRequest,
    auth: dict = Depends(require_tenant_permission("INGEST")),
):
    """Secured root alias for /documents/ingest."""
    return _ingest_document_endpoint(req=req, auth=auth)
