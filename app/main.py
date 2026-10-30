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
from app.services.vector_store import ingest_chunks, search_chunks
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


@app.get("/")
def home():
    return {"message": "PDF RAG API is running"}


@app.get("/auth/me")
def get_my_profile(user: dict = Depends(get_current_user)):
    return {"status": "authenticated", "user": user}


@app.post("/ingest")
def ingest_pdf(user: dict = Depends(get_current_user)):
    try:
        tenant_id = user.get("uid") or "default_user"
        pdf_folder = PDF_PATH.parent
        pdf_files = sorted(list(pdf_folder.glob("*.pdf"))) if pdf_folder.exists() else [PDF_PATH]

        total_chunks = 0
        file_summaries = []

        for pdf_file in pdf_files:
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
                source=pdf_file.name,
                vectors=vectors,
                tenant_id=tenant_id,
            )
            total_chunks += count
            file_summaries.append(f"{pdf_file.name} ({count} chunks)")

        return {
            "status": "success",
            "pdf": ", ".join(file_summaries),
            "chunks_ingested": total_chunks,
            "total_files": len(pdf_files),
            "tenant_id": tenant_id,
            "embedding_model": "gemini-embedding-001",
            "requested_by": user.get("email") or user.get("uid"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/sources")
def get_sources():
    """Returns the list of PDF documents available."""
    pdf_folder = PDF_PATH.parent
    if pdf_folder.exists():
        files = sorted([f.name for f in pdf_folder.glob("*.pdf")])
        return {"sources": files}
    return {"sources": [PDF_PATH.name]}


@app.post("/ask")
def ask_question(request: QuestionRequest, user: dict = Depends(get_current_user)):
    try:
        tenant_id = user.get("uid") or "default_user"
        return answer_question(
            request.question,
            tenant_id=tenant_id,
            source_filter=request.source_filter,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/ask-stream")
def ask_question_stream(request: QuestionRequest, user: dict = Depends(get_current_user)):
    tenant_id = user.get("uid") or "default_user"

    def event_generator():
        try:
            # 1. Check for greetings / non-document general conversation
            if is_conversational_or_greeting(request.question):
                # Send empty metadata (0 chunks, 0 reranked)
                empty_meta = json.dumps({"type": "metadata", "chunks": [], "reranked": []})
                yield f"data: {empty_meta}\n\n"

                # Direct conversational response without RAG
                greeting_prompt = f"""You are a helpful and polite document assistant.
Respond naturally, politely and concisely to the following greeting or general question:
"{request.question}"
Remind the user briefly that they can ask questions about their uploaded PDF documents."""

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

            # 2. Document Question: Retrieve chunks & Rerank
            q_vec = embed_query(request.question)
            chunks = search_chunks(
                query_vector=q_vec,
                tenant_id=tenant_id,
                source_filter=request.source_filter,
                limit=TOP_K,
            )
            reranked = rerank_chunks(question=request.question, chunks=chunks, top_k=TOP_K)

            meta_payload = json.dumps({
                "type": "metadata",
                "chunks": chunks,
                "reranked": reranked,
            })
            yield f"data: {meta_payload}\n\n"

            if not chunks:
                empty_payload = json.dumps({
                    "type": "token",
                    "token": "I could not find relevant information in the ingested document.",
                })
                yield f"data: {empty_payload}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            context = "\n\n".join([
                f"[source:{c['source']},page:{c['page']}]\n{c['content']}"
                for c in (reranked or chunks)
            ])
            prompt = f"""You are a document question answering assistant.
Context:
{context}

Question:
{request.question}
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








