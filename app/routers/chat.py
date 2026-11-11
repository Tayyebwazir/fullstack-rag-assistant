"""Question answering and SSE streaming RAG routes with strict tenant isolation."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from google import genai

from app.config import GEMINI_API_KEY, GENERATION_MODEL, TOP_K
from app.dependencies import require_tenant_permission, resolve_authorized_user
from app.schemas import QuestionRequest
from app.services.auth import get_current_user
from app.services.embeddings import embed_query
from app.services.intent import is_conversational_or_greeting
from app.services.rag_service import answer_question
from app.services.vector_store import search_chunks

try:
    from app.services.reranker import rerank_chunks
except (ImportError, AttributeError):
    def rerank_chunks(question="", chunks=None, top_k=None):
        return []

router = APIRouter(tags=["Q&A / RAG"])

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _sse_error_response(error_text: str) -> StreamingResponse:
    """Returns a single-event SSE error stream for authorization failures."""

    def _stream():
        payload = json.dumps({"type": "error", "error": error_text})
        yield f"data: {payload}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/ask")
@router.post("/question/ask")
def ask_question(
    request: QuestionRequest,
    auth: dict = Depends(require_tenant_permission("READ")),
):
    """Answers a user question strictly within their authorized Weaviate tenant shard."""
    authorized_tenant = auth["tenant_id"]
    try:
        return answer_question(
            request.question,
            tenant_id=authorized_tenant,
            source_filter=request.source_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ask-stream")
def ask_question_stream(
    request: QuestionRequest,
    user: dict = Depends(get_current_user),
):
    """Streams RAG answers as Server-Sent Events strictly scoped to the authorized tenant."""
    user_record, authorized_tenant, err_msg = resolve_authorized_user(user, "READ")
    if not user_record:
        return _sse_error_response("403 Forbidden: User ID not registered in database.")
    if err_msg:
        return _sse_error_response(f"403 Forbidden - {err_msg}")

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
                c["content"].strip()
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
