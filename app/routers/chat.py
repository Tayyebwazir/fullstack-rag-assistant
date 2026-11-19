"""Question answering and SSE streaming RAG routes with strict tenant isolation."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.config import GENERATION_MODEL
from app.dependencies import require_tenant_permission, resolve_authorized_user
from app.schemas import QuestionRequest
from app.services.auth import get_current_user
from app.services.gemini_client import get_gemini_client
from app.services.intent import is_conversational_or_greeting
from app.services.rag_service import answer_question, get_greeting_prompt, prepare_rag_pipeline

router = APIRouter(tags=["Q&A / RAG"])


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
    if request.tenant_id and request.tenant_id != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: You are not authorized to access tenant '{request.tenant_id}'.",
        )

    try:
        return answer_question(
            request.question,
            tenant_id=authorized_tenant,
            source_filter=request.source_filter,
        )
    except Exception as exc:
        logger.error("Error answering question: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process question due to an internal error.",
        ) from exc


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

    if request.tenant_id and request.tenant_id != authorized_tenant:
        return _sse_error_response(f"403 Forbidden - You are not authorized to access tenant '{request.tenant_id}'.")

    def event_generator():
        try:
            gemini_client = get_gemini_client()

            # 1. Check for greetings / non-document general conversation
            if is_conversational_or_greeting(request.question):
                empty_meta = json.dumps({"type": "metadata", "chunks": [], "reranked": []})
                yield f"data: {empty_meta}\n\n"

                greeting_prompt = get_greeting_prompt(request.question, tenant_id=authorized_tenant)
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

            # 2. Document Question: Use shared RAG pipeline
            pipeline_result = prepare_rag_pipeline(
                question=request.question,
                tenant_id=authorized_tenant,
                source_filter=request.source_filter,
            )

            meta_payload = json.dumps({
                "type": "metadata",
                "chunks": pipeline_result["chunks"],
                "reranked": pipeline_result["reranked"],
                "tenant": authorized_tenant,
            })
            yield f"data: {meta_payload}\n\n"

            if not pipeline_result["has_chunks"]:
                empty_payload = json.dumps({
                    "type": "token",
                    "token": f"No relevant information found in tenant [{authorized_tenant}].",
                })
                yield f"data: {empty_payload}\n\n"
                yield 'data: {"type": "done"}\n\n'
                return

            stream = gemini_client.models.generate_content_stream(
                model=GENERATION_MODEL,
                contents=pipeline_result["prompt"],
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
            logger.error("Error during streaming response: %s", err, exc_info=True)
            err_payload = json.dumps({
                "type": "error",
                "error": "An error occurred while generating the answer. Please try again.",
            })
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
