from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field

from app.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GEMINI_API_KEY,
    GENERATION_MODEL,
)
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_document, embed_documents, embed_query
from app.services.vector_store import (
    ingest_chunks,
    search_chunks,
    ensure_collection,
    connect_weaviate,
    VALID_TENANTS,
    TENANT_DOCUMENTS,
    TENANT_METADATA,
)
from app.services.auth import get_current_user
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
    title="Gemini + Weaviate PDF RAG API",
    version="1.0.0",
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


@app.get("/")
def home():
    return {"message": "PDF RAG API is running"}


@app.get("/auth/me")
def get_my_profile(user: dict = Depends(get_current_user)):
    return {"status": "authenticated", "user": user}


@app.get("/tenants")
def get_tenants(user: dict = Depends(get_current_user)):
    """Returns the three isolated multi-tenant workspaces: alpha, beta, and tri."""
    return {"tenants": TENANT_METADATA}


@app.post("/ingest")
def ingest_pdf(tenant_id: str | None = None, user: dict = Depends(get_current_user)):
    try:
        target_tenants = [tenant_id] if tenant_id in TENANT_DOCUMENTS else list(TENANT_DOCUMENTS.keys())
        pdf_folder = PDF_PATH.parent

        total_chunks = 0
        file_summaries = []

        for t in target_tenants:
            doc_name = TENANT_DOCUMENTS[t]
            pdf_file = pdf_folder / doc_name
            if not pdf_file.exists():
                continue

            pages = load_pdf(pdf_file)
            chunks = chunk_pages(
                pages,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
            if not chunks:
                continue

            contents = [chunk["content"] for chunk in chunks]
            vectors = embed_documents(contents)

            count = ingest_chunks(
                chunks=chunks,
                source=doc_name,
                vectors=vectors,
                tenant_id=t,
            )
            total_chunks += count
            file_summaries.append(f"{t}: {doc_name} ({count} chunks)")

        return {
            "status": "success",
            "tenants_ingested": file_summaries,
            "chunks_ingested": total_chunks,
            "requested_by": user.get("email") or user.get("uid"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/sources")
def get_sources(tenant_id: str | None = None):
    """Returns the PDF document associated with the selected tenant or all."""
    if tenant_id and tenant_id in TENANT_DOCUMENTS:
        return {"sources": [TENANT_DOCUMENTS[tenant_id]]}
    return {"sources": list(TENANT_DOCUMENTS.values())}


@app.post("/ask")
def ask_question(request: QuestionRequest, user: dict = Depends(get_current_user)):
    try:
        active_tenant = request.tenant_id if request.tenant_id in VALID_TENANTS else "alpha"
        return answer_question(
            request.question,
            tenant_id=active_tenant,
            source_filter=request.source_filter,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/ask-stream")
def ask_question_stream(request: QuestionRequest, user: dict = Depends(get_current_user)):
    active_tenant = request.tenant_id if request.tenant_id in VALID_TENANTS else "alpha"

    def event_generator():
        try:
            # 1. Check for greetings / non-document general conversation
            if is_conversational_or_greeting(request.question):
                empty_meta = json.dumps({"type": "metadata", "chunks": [], "reranked": []})
                yield f"data: {empty_meta}\n\n"

                greeting_prompt = f"""You are a helpful and polite document assistant.
Respond naturally, politely and concisely to the following greeting or general question:
"{request.question}"
Remind the user briefly that they can ask questions about their tenant's documents."""

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
                tenant_id=active_tenant,
                source_filter=request.source_filter,
                limit=TOP_K,
            )
            reranked = rerank_chunks(question=request.question, chunks=chunks, top_k=TOP_K)

            meta_payload = json.dumps({
                "type": "metadata",
                "chunks": chunks,
                "reranked": reranked,
                "tenant": active_tenant,
            })
            yield f"data: {meta_payload}\n\n"

            if not chunks:
                empty_payload = json.dumps({
                    "type": "token",
                    "token": f"No relevant information found in tenant [{active_tenant}].",
                })
                yield f"data: {empty_payload}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            context_blocks = [
                c['content'].strip()
                for c in (reranked or chunks)
            ]
            context = "\n\n---\n\n".join(context_blocks)

            prompt = f"""You are an expert AI document assistant for tenant [{active_tenant}].
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
