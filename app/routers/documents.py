"""Tenant-isolated document management, OCR extraction, and ingestion routes."""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INGEST_RATE_LIMIT,
    INGEST_RATE_WINDOW_SECONDS,
    MAX_CHUNKS_PER_DOC,
    MAX_PDF_PAGES,
)
from app.dependencies import (
    get_authorized_document,
    require_tenant_permission,
    resolve_authorized_user,
)
from app.schemas import DocIdRequest
from app.services.auth import get_current_user
from app.services.chunker import chunk_pages
from app.services.document_loader import load_document
from app.services.embeddings import embed_documents
from app.services.firebase_db import (
    create_document_record,
    delete_document_record,
    get_tenant_documents,
    save_document_chunks,
    update_document_record,
)
from app.services.vector_store import (
    delete_tenant_document,
    delete_tenant_document_by_id,
    get_tenant_documents_summary,
    ingest_chunks,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


async def _validate_and_read_upload(file: UploadFile, empty_detail: str) -> tuple[str, bytes]:
    """Validates uploaded file extension and size, returning `(ext, bytes)`."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, DOC.",
        )

    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(status_code=400, detail=empty_detail)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 25MB limit.")
    return ext, contents


@router.get("/list")
def list_documents(
    tenant_id: str | None = None,
    auth: dict = Depends(require_tenant_permission("READ")),
):
    """Lists only the documents belonging to the authenticated user's authorized tenant."""
    authorized_tenant = auth["tenant_id"]
    if tenant_id and tenant_id != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: You are not authorized to access tenant '{tenant_id}'.",
        )
    tenant_docs = get_tenant_documents(authorized_tenant)
    return {
        "status": "success",
        "tenant_id": authorized_tenant,
        "documents": tenant_docs,
    }

@router.get("/sources")
def get_sources(
    tenant_id: str | None = None,
    user: dict = Depends(get_current_user),
):
    """Returns document sources strictly isolated to the authenticated user's authorized tenant."""
    if not user or not user.get("uid"):
        raise HTTPException(
            status_code=401,
            detail="Authentication required: missing or invalid user UID.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_record, authorized_tenant, err_msg = resolve_authorized_user(user, "READ")
    if not user_record or not authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=err_msg or "Access Denied: No authorized tenant assigned.",
        )

    # Prevent cross-tenant data exposure
    if tenant_id and tenant_id != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: You are not authorized to access tenant '{tenant_id}'.",
        )

    weaviate_summary = get_tenant_documents_summary(authorized_tenant)
    weaviate_sources = {item["source"] for item in weaviate_summary if item.get("source")}
    firestore_docs = get_tenant_documents(authorized_tenant)
    firestore_sources = {doc.get("filename") for doc in firestore_docs if doc.get("filename")}

    sources = sorted(list(weaviate_sources | firestore_sources))
    return {"sources": sources}

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    auth: dict = Depends(require_tenant_permission("WRITE")),
):
    """Uploads a new PDF, DOCX, DOC, or image file for the authenticated user's authorized tenant."""
    user = auth["user"]
    user_record = auth["user_record"]
    authorized_tenant = auth["tenant_id"]
    uid = user.get("uid")

    ext, contents = await _validate_and_read_upload(file, "Uploaded file is empty.")

    doc_id = str(uuid.uuid4())
    safe_filename = Path(file.filename).name.replace(" ", "_")
    tenant_upload_dir = BASE_DIR / "app" / "data" / "tenants" / authorized_tenant
    tenant_upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = tenant_upload_dir / f"{doc_id}_{safe_filename}"

    with open(save_path, "wb") as f:
        f.write(contents)

    now_iso = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(contents).hexdigest()

    doc_record = {
        "doc_id": doc_id,
        "tenant_id": authorized_tenant,
        "owner_uid": uid,
        "owner_email": user_record.get("email") or user.get("email", ""),
        "filename": file.filename,
        "file_type": ext.replace(".", "").upper(),
        "file_path": str(save_path),
        "file_size": len(contents),
        "content_hash": content_hash,
        "upload_timestamp": now_iso,
        "updated_timestamp": now_iso,
        "version": 1,
        "status": "uploaded",
        "chunk_count": 0,
        "error_message": None,
    }

    created = create_document_record(doc_record, user.get("id_token"))
    return {
        "status": "success",
        "message": f"Document '{file.filename}' uploaded successfully for Tenant '{authorized_tenant}'.",
        "document": created,
    }


RATE_LIMIT_STORE = BASE_DIR / "app" / "data" / "rate_limits.json"


def _compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 content hash of the document file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_ingest_rate_limit(uid: str) -> None:
    """Enforces shared rate limit per user per hour across instances, returning 429 if exceeded."""
    RATE_LIMIT_STORE.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    records = {}
    if RATE_LIMIT_STORE.exists():
        try:
            with open(RATE_LIMIT_STORE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = {}

    user_timestamps = records.get(uid, [])
    # Filter to timestamps within the rolling window (last 1 hour)
    valid_timestamps = [ts for ts in user_timestamps if (now - ts) < INGEST_RATE_WINDOW_SECONDS]

    if len(valid_timestamps) >= INGEST_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: Maximum {INGEST_RATE_LIMIT} document ingestions per hour. Please try again later.",
        )

    valid_timestamps.append(now)
    records[uid] = valid_timestamps
    try:
        with open(RATE_LIMIT_STORE, "w", encoding="utf-8") as f:
            json.dump(records, f)
    except Exception as e:
        logger.warning("Could not persist rate limits: %s", e)


@router.post("/ingest")
def ingest_document(
    req: DocIdRequest,
    auth: dict = Depends(require_tenant_permission("INGEST")),
):
    """Securely chunks, embeds, and ingests a document with rate limiting and duplicate detection."""
    user = auth["user"]
    uid = user.get("uid")
    authorized_tenant = auth["tenant_id"]

    # 1. Enforce rate limiting: 5 requests per user per hour -> 429
    check_ingest_rate_limit(uid)

    # 2. Prevent cross-tenant ingestion -> 403
    if req.tenant_id and req.tenant_id != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: You are not authorized to ingest documents into tenant '{req.tenant_id}'.",
        )

    # 3. Verify document ownership within authorized tenant
    doc = get_authorized_document(req.doc_id, authorized_tenant)
    file_path = Path(doc.get("file_path", ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Underlying document file not found on server.")

    # 4. Duplicate PDF detection using content hash across the tenant
    file_hash = _compute_file_hash(file_path)

    # Check A: Has THIS exact doc record already been completed?
    if doc.get("status") == "completed" and doc.get("chunk_count", 0) > 0:
        return {
            "status": "success",
            "message": f"Document '{doc.get('filename')}' is already ingested with {doc.get('chunk_count')} chunks. Duplicate processing skipped.",
            "chunks_ingested": doc.get("chunk_count", 0),
            "document": doc,
            "filename": doc.get("filename", file_path.name),
            "skipped_duplicate": True,
        }

    # Check B: Has ANY document in this tenant already been ingested with the same content hash or filename+size?
    tenant_docs = get_tenant_documents(authorized_tenant)
    for existing_doc in tenant_docs:
        if (
            existing_doc.get("doc_id") != req.doc_id
            and existing_doc.get("status") == "completed"
            and existing_doc.get("chunk_count", 0) > 0
            and (
                existing_doc.get("content_hash") == file_hash
                or (
                    existing_doc.get("filename") == doc.get("filename")
                    and existing_doc.get("file_size") == doc.get("file_size")
                )
            )
        ):
            # Same content already exists in tenant's Weaviate shard! Link count to avoid duplicate API calls
            existing_chunks = existing_doc.get("chunk_count", 0)
            updated_doc = update_document_record(
                req.doc_id,
                {
                    "status": "completed",
                    "chunk_count": existing_chunks,
                    "content_hash": file_hash,
                    "error_message": None,
                    "updated_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                user.get("id_token"),
            )
            return {
                "status": "success",
                "message": f"Duplicate document detected! '{doc.get('filename')}' matches previously ingested '{existing_doc.get('filename')}' ({existing_chunks} chunks already in shard). Re-ingestion skipped.",
                "chunks_ingested": existing_chunks,
                "document": updated_doc or doc,
                "filename": doc.get("filename", file_path.name),
                "skipped_duplicate": True,
            }

    update_document_record(req.doc_id, {"status": "processing", "error_message": None}, user.get("id_token"))

    try:
        # 5. Extract pages & enforce page-count limit (safeguard against excessive Gemini costs)
        pages = load_document(file_path)
        if len(pages) > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Document exceeds maximum limit of {MAX_PDF_PAGES} pages (uploaded: {len(pages)} pages).",
            )

        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text found to chunk in the document.")

        if len(chunks) > MAX_CHUNKS_PER_DOC:
            raise HTTPException(
                status_code=400,
                detail=f"Document produced {len(chunks)} chunks, exceeding the safety limit of {MAX_CHUNKS_PER_DOC} chunks.",
            )

        contents = [c["content"] for c in chunks]
        vectors = embed_documents(contents)

        count = ingest_chunks(
            chunks=chunks,
            source=doc.get("filename", file_path.name),
            vectors=vectors,
            tenant_id=authorized_tenant,
            doc_id=doc.get("doc_id", ""),
        )

        save_document_chunks(
            doc_id=req.doc_id,
            chunks=chunks,
            tenant_id=authorized_tenant,
            source=doc.get("filename", file_path.name),
            id_token=user.get("id_token"),
        )

        updated_doc = update_document_record(
            req.doc_id,
            {
                "status": "completed",
                "chunk_count": count,
                "content_hash": file_hash,
                "error_message": None,
                "updated_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            user.get("id_token"),
        )

        full_extracted_text = "\n\n".join(
            f"[Page {p['page']}]\n{p['text']}" for p in pages if p.get("text", "").strip()
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {count} chunks of '{doc.get('filename')}' into Tenant '{authorized_tenant}'.",
            "chunks_ingested": count,
            "document": updated_doc,
            "extracted_text": full_extracted_text,
            "filename": doc.get("filename", file_path.name),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document ingestion failed for doc_id '%s': %s", req.doc_id, exc, exc_info=True)
        update_document_record(
            req.doc_id,
            {"status": "failed", "error_message": "Internal ingestion error."},
            user.get("id_token"),
        )
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed due to an internal processing error.",
        ) from exc


@router.put("/update")
async def update_document(
    doc_id: str = Form(...),
    file: UploadFile = File(...),
    auth: dict = Depends(require_tenant_permission("UPDATE")),
):
    """Replaces an existing document with a new version, re-extracts, re-chunks,
    and updates vector embeddings in the authorized tenant's shard safely.
    """
    user = auth["user"]
    authorized_tenant = auth["tenant_id"]
    doc = get_authorized_document(doc_id, authorized_tenant)

    ext, contents = await _validate_and_read_upload(file, "Replacement file is empty.")

    new_version = doc.get("version", 1) + 1
    safe_filename = Path(file.filename).name.replace(" ", "_")
    tenant_upload_dir = BASE_DIR / "app" / "data" / "tenants" / authorized_tenant
    tenant_upload_dir.mkdir(parents=True, exist_ok=True)
    staged_path = tenant_upload_dir / f"{doc_id}_v{new_version}_{safe_filename}"

    with open(staged_path, "wb") as f:
        f.write(contents)

    try:
        pages = load_document(staged_path)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunk_contents = [c["content"] for c in chunks]
        vectors = embed_documents(chunk_contents)

        delete_tenant_document_by_id(doc_id, authorized_tenant)

        count = ingest_chunks(
            chunks=chunks,
            source=file.filename,
            vectors=vectors,
            tenant_id=authorized_tenant,
            doc_id=doc_id,
        )

        save_document_chunks(
            doc_id=doc_id,
            chunks=chunks,
            tenant_id=authorized_tenant,
            source=file.filename,
            id_token=user.get("id_token"),
        )

        old_path = Path(doc.get("file_path", ""))
        if old_path.exists() and old_path != staged_path:
            try:
                old_path.unlink()
            except Exception:
                pass

        now_iso = datetime.now(timezone.utc).isoformat()
        updated_doc = update_document_record(
            doc_id,
            {
                "filename": file.filename,
                "file_type": ext.replace(".", "").upper(),
                "file_path": str(staged_path),
                "file_size": len(contents),
                "version": new_version,
                "status": "completed",
                "chunk_count": count,
                "updated_timestamp": now_iso,
                "error_message": None,
            },
            user.get("id_token"),
        )

        return {
            "status": "success",
            "message": f"Document updated to version {new_version} and re-ingested with {count} chunks in Tenant '{authorized_tenant}'.",
            "document": updated_doc,
        }
    except Exception as exc:
        if staged_path.exists():
            try:
                staged_path.unlink()
            except Exception:
                pass
        logger.error("Document update failed for doc_id '%s': %s", doc_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Document update failed due to an internal error. Original document remained safe.",
        ) from exc


@router.delete("/delete")
def delete_document(
    req: DocIdRequest,
    auth: dict = Depends(require_tenant_permission("DELETE")),
):
    """Permanently deletes a document belonging to the authenticated tenant."""
    user = auth["user"]
    authorized_tenant = auth["tenant_id"]
    doc = get_authorized_document(req.doc_id, authorized_tenant)

    delete_tenant_document_by_id(doc.get("doc_id", ""), authorized_tenant)
    delete_tenant_document(doc.get("filename", ""), authorized_tenant)

    file_path = Path(doc.get("file_path", ""))
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            print(f"[FileStorage] Deletion note: {e}")

    delete_document_record(req.doc_id, user.get("id_token"))

    return {
        "status": "success",
        "message": f"Permanently deleted document '{doc.get('filename')}' from Tenant '{authorized_tenant}'.",
        "doc_id": req.doc_id,
        "tenant_id": authorized_tenant,
    }


@router.get("/{doc_id}/extracted-text")
def get_extracted_text(
    doc_id: str,
    auth: dict = Depends(require_tenant_permission("READ")),
):
    """Returns the exact extracted text from an uploaded image, note, or PDF."""
    authorized_tenant = auth["tenant_id"]
    doc = get_authorized_document(doc_id, authorized_tenant)

    file_path = Path(doc.get("file_path", ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Underlying document file not found on server.")

    pages = load_document(file_path)
    full_text = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages if p.get("text", "").strip())
    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": doc.get("filename", file_path.name),
        "file_type": doc.get("file_type", ""),
        "total_pages": len(pages),
        "extracted_text": full_text,
    }
