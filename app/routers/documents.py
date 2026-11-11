"""Tenant-isolated document management, OCR extraction, and ingestion routes."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from app.dependencies import get_authorized_document, require_tenant_permission
from app.schemas import DocIdRequest
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
def list_documents(auth: dict = Depends(require_tenant_permission("READ"))):
    """Lists only the documents belonging to the authenticated user's authorized tenant."""
    authorized_tenant = auth["tenant_id"]
    tenant_docs = get_tenant_documents(authorized_tenant)
    return {
        "status": "success",
        "tenant_id": authorized_tenant,
        "documents": tenant_docs,
    }


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
    doc_record = {
        "doc_id": doc_id,
        "tenant_id": authorized_tenant,
        "owner_uid": uid,
        "owner_email": user_record.get("email") or user.get("email", ""),
        "filename": file.filename,
        "file_type": ext.replace(".", "").upper(),
        "file_path": str(save_path),
        "file_size": len(contents),
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


@router.post("/ingest")
def ingest_document(
    req: DocIdRequest,
    auth: dict = Depends(require_tenant_permission("INGEST")),
):
    """Chunks, embeds, and ingests an uploaded document into the authorized tenant's Weaviate shard."""
    user = auth["user"]
    authorized_tenant = auth["tenant_id"]
    doc = get_authorized_document(req.doc_id, authorized_tenant)

    file_path = Path(doc.get("file_path", ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Underlying document file not found on server.")

    update_document_record(req.doc_id, {"status": "processing", "error_message": None}, user.get("id_token"))

    try:
        pages = load_document(file_path)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("No readable text found to chunk in the document.")

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
    except Exception as exc:
        update_document_record(
            req.doc_id,
            {"status": "failed", "error_message": str(exc)},
            user.get("id_token"),
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


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
        raise HTTPException(
            status_code=500,
            detail=f"Update failed. Original document remained safe: {exc}",
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
