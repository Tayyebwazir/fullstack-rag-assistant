from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import shutil
from pathlib import Path
from pydantic import BaseModel, Field

from app.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GEMINI_API_KEY,
    GENERATION_MODEL,
    BASE_DIR,
)
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_document, embed_documents, embed_query
from app.services.vector_store import (
    ingest_chunks,
    search_chunks,
    delete_tenant_document,
    get_tenant_documents_summary,
    ensure_collection,
    connect_weaviate,
    VALID_TENANTS,
    TENANT_DOCUMENTS,
    TENANT_METADATA,
)
from app.services.auth import get_current_user
from app.services.firebase_db import (
    get_user_record,
    update_user_tenant_and_permissions,
    verify_tenant_access,
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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source_filter: str | None = Field(default=None, description="Optional PDF document filter")
    tenant_id: str | None = Field(default="alpha", description="Target multi-tenant workspace (alpha, beta, tri)")


class SelectTenantRequest(BaseModel):
    tenant_id: str


class DocOperationRequest(BaseModel):
    source: str
    tenant_id: str


@app.get("/")
def home():
    return {"message": "Multi-Tenant RAG Assistant API is running"}


# ==========================================
# 1. User & Firebase Tenant Authorisation
# ==========================================

@app.get("/user/profile")
def get_user_profile(user: dict = Depends(get_current_user)):
    """Fetches user UUID, assigned tenant ID, and permissions from Firebase Database."""
    uid = user.get("uid") or "default_user"
    email = user.get("email") or ""
    profile = get_user_record(uid, email)
    return {
        "status": "success",
        "profile": profile,
        "available_tenants": TENANT_METADATA,
    }


@app.post("/user/select-tenant")
def select_user_tenant(req: SelectTenantRequest, user: dict = Depends(get_current_user)):
    """Allows the user to select/switch their active tenant in Firebase Database."""
    uid = user.get("uid") or "default_user"
    if req.tenant_id not in VALID_TENANTS:
        raise HTTPException(status_code=400, detail=f"Invalid tenant. Must be one of: {VALID_TENANTS}")
    updated = update_user_tenant_and_permissions(uid=uid, tenant_id=req.tenant_id)
    return {"status": "success", "profile": updated}


@app.get("/tenants")
def get_tenants(user: dict = Depends(get_current_user)):
    """Returns the three isolated multi-tenant workspaces: alpha, beta, and tri."""
    return {"tenants": TENANT_METADATA}


# ==========================================
# 2. Document Operations (Upload, Ingest, Update, Delete)
# ==========================================

@app.get("/documents/list")
def list_documents(tenant_id: str = "alpha", user: dict = Depends(get_current_user)):
    """Lists documents ingested into this tenant's shard."""
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))
    
    # Verify tenant authorisation
    authorized, err_msg = verify_tenant_access(user_record, tenant_id, "read")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    weaviate_docs = get_tenant_documents_summary(tenant_id)
    pdf_folder = BASE_DIR / "app" / "data"
    local_files = [f.name for f in pdf_folder.glob("*.pdf")]

    return {
        "tenant": tenant_id,
        "weaviate_documents": weaviate_docs,
        "available_files": local_files,
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form("alpha"),
    user: dict = Depends(get_current_user),
):
    """Uploads a new document for the authorized tenant."""
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))

    # Verify user belongs to tenant and has 'write' permission
    authorized, err_msg = verify_tenant_access(user_record, tenant_id, "write")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = BASE_DIR / "app" / "data" / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "message": f"Document '{file.filename}' uploaded for tenant '{tenant_id}'. Ready for ingestion.",
        "filename": file.filename,
        "tenant_id": tenant_id,
    }


@app.post("/documents/ingest")
def ingest_document(req: DocOperationRequest, user: dict = Depends(get_current_user)):
    """Chunks, embeds, and ingests a document into the authorized tenant's shard."""
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))

    # Verify user belongs to tenant and has 'ingest' permission
    authorized, err_msg = verify_tenant_access(user_record, req.tenant_id, "ingest")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    pdf_file = BASE_DIR / "app" / "data" / req.source
    if not pdf_file.exists():
        raise HTTPException(status_code=404, detail=f"File '{req.source}' not found on server.")

    try:
        pages = load_pdf(pdf_file)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text found in document.")

        contents = [c["content"] for c in chunks]
        vectors = embed_documents(contents)

        count = ingest_chunks(
            chunks=chunks,
            source=pdf_file.name,
            vectors=vectors,
            tenant_id=req.tenant_id,
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {count} chunks of '{req.source}' into tenant '{req.tenant_id}'!",
            "chunks_ingested": count,
            "source": req.source,
            "tenant_id": req.tenant_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/documents/update")
async def update_document(
    file: UploadFile = File(...),
    tenant_id: str = Form("alpha"),
    user: dict = Depends(get_current_user),
):
    """Updates an existing document and re-ingests vectors into the tenant shard."""
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))

    # Verify user belongs to tenant and has 'update' permission
    authorized, err_msg = verify_tenant_access(user_record, tenant_id, "update")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    # Overwrite file
    save_path = BASE_DIR / "app" / "data" / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Re-embed and update Weaviate tenant shard
    pages = load_pdf(save_path)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    contents = [c["content"] for c in chunks]
    vectors = embed_documents(contents)

    count = ingest_chunks(
        chunks=chunks,
        source=file.filename,
        vectors=vectors,
        tenant_id=tenant_id,
    )

    return {
        "status": "success",
        "message": f"Document '{file.filename}' updated and re-ingested with {count} chunks in tenant '{tenant_id}'.",
        "chunks_ingested": count,
        "tenant_id": tenant_id,
    }


@app.delete("/documents/delete")
def delete_document(req: DocOperationRequest, user: dict = Depends(get_current_user)):
    """Deletes all chunks of a document from the tenant's Weaviate shard."""
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))

    # Verify user belongs to tenant and has 'delete' permission
    authorized, err_msg = verify_tenant_access(user_record, req.tenant_id, "delete")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    deleted_count = delete_tenant_document(source_name=req.source, tenant_id=req.tenant_id)

    return {
        "status": "success",
        "message": f"Deleted document '{req.source}' from tenant '{req.tenant_id}' shard.",
        "tenant_id": req.tenant_id,
        "source": req.source,
    }


@app.get("/sources")
def get_sources(tenant_id: str | None = None):
    """Returns the document associated with the selected tenant or all."""
    if tenant_id and tenant_id in TENANT_DOCUMENTS:
        return {"sources": [TENANT_DOCUMENTS[tenant_id]]}
    return {"sources": list(TENANT_DOCUMENTS.values())}


# ==========================================
# 3. Query & SSE Streaming (With Tenant Authorisation)
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest, user: dict = Depends(get_current_user)):
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))
    target_tenant = request.tenant_id if request.tenant_id in VALID_TENANTS else "alpha"

    # Enforce tenant authorisation for reading
    authorized, err_msg = verify_tenant_access(user_record, target_tenant, "read")
    if not authorized:
        raise HTTPException(status_code=403, detail=err_msg)

    try:
        return answer_question(
            request.question,
            tenant_id=target_tenant,
            source_filter=request.source_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-stream")
def ask_question_stream(request: QuestionRequest, user: dict = Depends(get_current_user)):
    uid = user.get("uid") or "default_user"
    user_record = get_user_record(uid, user.get("email", ""))
    target_tenant = request.tenant_id if request.tenant_id in VALID_TENANTS else "alpha"

    # Enforce tenant authorisation
    authorized, err_msg = verify_tenant_access(user_record, target_tenant, "read")
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

                greeting_prompt = f"""You are a helpful and polite document assistant for tenant [{target_tenant}].
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

            # 2. Document Question: Retrieve chunks strictly within active_tenant shard & Rerank
            q_vec = embed_query(request.question)
            chunks = search_chunks(
                query_vector=q_vec,
                tenant_id=target_tenant,
                source_filter=request.source_filter,
                limit=TOP_K,
            )
            reranked = rerank_chunks(question=request.question, chunks=chunks, top_k=TOP_K)

            meta_payload = json.dumps({
                "type": "metadata",
                "chunks": chunks,
                "reranked": reranked,
                "tenant": target_tenant,
            })
            yield f"data: {meta_payload}\n\n"

            if not chunks:
                empty_payload = json.dumps({
                    "type": "token",
                    "token": f"No relevant information found in tenant [{target_tenant}].",
                })
                yield f"data: {empty_payload}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            context_blocks = [
                c['content'].strip()
                for c in (reranked or chunks)
            ]
            context = "\n\n---\n\n".join(context_blocks)

            prompt = f"""You are an expert AI document assistant for tenant [{target_tenant}].
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

            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as err:
            err_payload = json.dumps({
                "type": "error",
                "error": str(err),
            })
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
