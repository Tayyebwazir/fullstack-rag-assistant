from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GEMINI_API_KEY,
    GENERATION_MODEL,
    BASE_DIR,
)
from app.services.document_loader import load_document
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_documents, embed_query
from app.services.vector_store import (
    ingest_chunks,
    search_chunks,
    delete_tenant_document_by_id,
    delete_tenant_document,
    get_tenant_documents_summary,
    ensure_collection,
    connect_weaviate,
)
from app.services.auth import get_current_user
from app.services.firebase_db import (
    get_user_record,
    verify_tenant_access,
    get_tenant_documents,
    get_document_by_id,
    create_document_record,
    update_document_record,
    delete_document_record,
)
from app.services.intent import is_conversational_or_greeting
try:
    from app.services.reranker import rerank_chunks
except (ImportError, AttributeError):
    def rerank_chunks(question="", chunks=None, top_k=None):
        return []
from app.services.rag_service import answer_question
from google import genai

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Multi-Tenant RAG Assistant API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source_filter: str | None = Field(default=None, description="Optional document filter")


class DocIdRequest(BaseModel):
    doc_id: str


@app.get("/")
def home():
    return {"message": "Multi-Tenant RAG Assistant API is running"}


# ==========================================
# 1. User & Firebase Tenant Authorisation
# ==========================================

@app.get("/user/profile")
def get_user_profile(user: dict = Depends(get_current_user)):
    """Fetches user UUID, assigned tenant ID, active status, and permissions from Firebase Database."""
    uid = user.get("uid")
    email = user.get("email", "")
    id_token = user.get("id_token")
    profile = get_user_record(uid, email, id_token)
    if not profile:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database. Please contact an administrator to assign you a tenant.",
        )
    if not profile.get("is_active", True):
        raise HTTPException(
            status_code=403,
            detail="Account Inactive: Your account has been suspended.",
        )
    return {
        "status": "success",
        "profile": profile,
        "assigned_tenant": profile.get("tenant_id", "alpha"),
    }


# ==========================================
# 2. Tenant Document Management & Operations
# ==========================================

@app.get("/documents/list")
def list_documents(user: dict = Depends(get_current_user)):
    """Lists only the documents belonging to the authenticated user's authorized tenant."""
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Verify READ permission on authorized tenant
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "READ")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    tenant_docs = get_tenant_documents(authorized_tenant)
    return {
        "status": "success",
        "tenant_id": authorized_tenant,
        "documents": tenant_docs,
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Uploads a new PDF, DOCX, or DOC file for the authenticated user's single authorized tenant.
    Resolves authorized tenant strictly on the backend from Firebase Database.
    """
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Verify WRITE permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "WRITE")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, DOC.",
        )

    # Read and validate file size
    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 25MB limit.")

    # Generate unique document UUID and save to tenant-specific isolated directory
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
        "filename": file.filename,
        "file_type": ext.replace(".", "").upper(),
        "file_path": str(save_path),
        "file_size": file_size,
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


@app.post("/documents/ingest")
def ingest_document(req: DocIdRequest, user: dict = Depends(get_current_user)):
    """Chunks, embeds, and ingests an uploaded document into the authorized tenant's Weaviate shard."""
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Verify INGEST permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "INGEST")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    doc = get_document_by_id(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{req.doc_id}' not found.")

    # Strictly enforce tenant ownership
    if doc.get("tenant_id") != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: Document '{req.doc_id}' belongs to another tenant.",
        )

    file_path = Path(doc.get("file_path", ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Underlying document file not found on server.")

    # Update status to processing
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

        return {
            "status": "success",
            "message": f"Successfully ingested {count} chunks of '{doc.get('filename')}' into Tenant '{authorized_tenant}'.",
            "chunks_ingested": count,
            "document": updated_doc,
        }
    except Exception as exc:
        update_document_record(
            req.doc_id,
            {"status": "failed", "error_message": str(exc)},
            user.get("id_token"),
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.put("/documents/update")
async def update_document(
    doc_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Replaces an existing document with a new version, re-extracts, re-chunks,
    and updates vector embeddings in the authorized tenant's shard safely.
    """
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Verify UPDATE permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "UPDATE")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")

    if doc.get("tenant_id") != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: Document '{doc_id}' belongs to another tenant.",
        )

    # Validate replacement file format
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, DOC.",
        )

    contents = await file.read()
    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Replacement file is empty.")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 25MB limit.")

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

        # Delete previous vectors for this doc_id
        delete_tenant_document_by_id(doc_id, authorized_tenant)

        # Ingest new chunks
        count = ingest_chunks(
            chunks=chunks,
            source=file.filename,
            vectors=vectors,
            tenant_id=authorized_tenant,
            doc_id=doc_id,
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
                "file_size": file_size,
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


@app.delete("/documents/delete")
def delete_document(req: DocIdRequest, user: dict = Depends(get_current_user)):
    """Permanently deletes a document belonging to the authenticated tenant."""
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Verify DELETE permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "DELETE")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    doc = get_document_by_id(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{req.doc_id}' not found.")

    if doc.get("tenant_id") != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: Document '{req.doc_id}' belongs to another tenant.",
        )

    # 1. Remove vector chunks from Weaviate
    delete_tenant_document_by_id(doc.get("doc_id", ""), authorized_tenant)
    delete_tenant_document(doc.get("filename", ""), authorized_tenant)

    # 2. Remove file from storage
    file_path = Path(doc.get("file_path", ""))
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            print(f"[FileStorage] Deletion note: {e}")

    # 3. Remove metadata record from Firebase DB
    delete_document_record(req.doc_id, user.get("id_token"))

    return {
        "status": "success",
        "message": f"Permanently deleted document '{doc.get('filename')}' from Tenant '{authorized_tenant}'.",
        "doc_id": req.doc_id,
        "tenant_id": authorized_tenant,
    }


# ==========================================
# 3. Query & SSE Streaming (Strict Tenant Isolation)
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest, user: dict = Depends(get_current_user)):
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database.",
        )
    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Enforce READ permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "READ")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    try:
        return answer_question(
            request.question,
            tenant_id=authorized_tenant,
            source_filter=request.source_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-stream")
def ask_question_stream(request: QuestionRequest, user: dict = Depends(get_current_user)):
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        def unauth_stream():
            payload = json.dumps({"type": "error", "error": "403 Forbidden: User ID not registered in database."})
            yield f"data: {payload}\n\n"
        return StreamingResponse(unauth_stream(), media_type="text/event-stream")

    authorized_tenant = user_record.get("tenant_id", "alpha")

    # Enforce READ permission
    authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, "READ")
    if not authorized:
        def auth_error_stream():
            payload = json.dumps({"type": "error", "error": f"403 Forbidden - {err_msg}"})
            yield f"data: {payload}\n\n"
        return StreamingResponse(auth_error_stream(), media_type="text/event-stream")

    def event_generator():
        try:
            # 1. Check for greetings / non-document general conversation
            if is_conversational_or_greeting(request.question):
                empty_meta = json.dumps({"type": "metadata", "chunks": [], "reranked": []})
                yield f"data: {empty_meta}\n\n"

                greeting_prompt = f"""You are a helpful and polite document assistant for tenant [{authorized_tenant}].
Respond naturally, politely and concisely to the following greeting or general question:
"{request.question}"
Remind the user briefly that they can ask questions about documents in their authorized tenant."""

                stream = gemini_client.models.generate_content_stream(
                    model=GENERATION_MODEL,
                    contents=greeting_prompt,
                )
                for chunk in stream:
                    if chunk.text:
                        token_payload = json.dumps({"type": "token", "token": chunk.text})
                        yield f"data: {token_payload}\n\n"

                yield 'data: {"type": "done"}\n\n'
                return

            # 2. Document Question: Retrieve chunks strictly within active authorized tenant shard & Rerank
            q_vec = embed_query(request.question)
            chunks = search_chunks(
                query_vector=q_vec,
                tenant_id=authorized_tenant,
                source_filter=request.source_filter,
                limit=TOP_K,
            )
            reranked = rerank_chunks(question=request.question, chunks=chunks, top_k=TOP_K)

            meta_payload = json.dumps({
                "type": "metadata",
                "chunks": chunks,
                "reranked": reranked,
                "tenant": authorized_tenant,
            })
            yield f"data: {meta_payload}\n\n"

            if not chunks:
                empty_payload = json.dumps({
                    "type": "token",
                    "token": f"No relevant information found in tenant [{authorized_tenant}].",
                })
                yield f"data: {empty_payload}\n\n"
                yield 'data: {"type": "done"}\n\n'
                return

            context_blocks = [
                c['content'].strip()
                for c in (reranked or chunks)
            ]
            context = "\n\n---\n\n".join(context_blocks)

            prompt = f"""You are an expert AI document assistant for tenant [{authorized_tenant}].
Provide a clear, detailed, comprehensive, and well-structured answer to the user's question based strictly on the provided context.

Context:
{context}

Question:
{request.question}

Guidelines:
- Deliver high quality, direct, well-organized explanations formatted with clean Markdown (use headings, bullet points, bold keywords, and code snippets when present).
- Do NOT output raw citation brackets or source tags (such as "[source:...]", "[page:...]") in your answer text. Document sources and page numbers are already presented in a dedicated visual panel.
- Explain technical definitions and include code examples if available in the text.
"""
            stream = gemini_client.models.generate_content_stream(
                model=GENERATION_MODEL,
                contents=prompt,
            )
            for chunk in stream:
                if chunk.text:
                    token_payload = json.dumps({
                        "type": "token",
                        "token": chunk.text,
                    })
                    yield f"data: {token_payload}\n\n"

            yield 'data: {"type": "done"}\n\n'

        except Exception as err:
            err_payload = json.dumps({
                "type": "error",
                "error": str(err),
            })
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
