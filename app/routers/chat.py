"""Question answering and SSE streaming RAG routes with strict tenant isolation."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from google import genai

from app.config import GEMINI_API_KEY, GENERATION_MODEL, RERANK_TOP_K, RETRIEVAL_TOP_K
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
SYSTEM_PROMPT = """You are a helpful student AI assistant.

Rules:
1. Answer questions ONLY using the provided context.
2. Do not use outside knowledge.
3. If the context does not contain the answer, say: "I don't know based on the provided documents."
4. Always cite the source document and page number.
5. Do not invent information or citations.
"""

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
                limit=RETRIEVAL_TOP_K,
            )
            reranked = rerank_chunks(
                question=request.question,
                chunks=chunks,
                top_k=RERANK_TOP_K,
            )

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
                f"[Document: {c.get('source', 'Unknown')} | Page: {c.get('page', 'N/A')}]\n{c.get('content', '').strip()}"
                for c in (reranked or chunks)
            ]
            context = "\n\n---\n\n".join(context_blocks)

            prompt = f"""{SYSTEM_PROMPT}

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

            yield 'data: {"type": "done"}\n\n'

        except Exception as err:
            err_payload = json.dumps({
                "type": "error",
                "error": str(err),
            })
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
