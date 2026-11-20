from app.config import (
    GENERATION_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from app.services.embeddings import embed_query
from app.services.vector_store import search_chunks
from app.services.intent import is_conversational_or_greeting
from app.services.reranker import rerank_chunks
from app.services.gemini_client import get_gemini_client

SYSTEM_PROMPT = """You are a helpful student AI assistant.

Rules:
1. Answer questions ONLY using the provided context.
2. Do not use outside knowledge.
3. If the context does not contain the answer, say: "I don't know based on the provided documents."
4. Always cite the source document and page number.
5. Do not invent information or citations.
"""


def get_greeting_prompt(question: str, tenant_id: str | None = None) -> str:
    """Shared greeting prompt for both standard and streaming responses."""
    tenant_info = f" for tenant [{tenant_id}]" if tenant_id else ""
    return f"""You are a helpful and polite document assistant{tenant_info}.
Respond naturally, politely and concisely to the following greeting or general question:
"{question}"
Remind the user briefly that they can ask questions about their uploaded PDF documents."""


def prepare_rag_pipeline(
    question: str,
    tenant_id: str,
    source_filter: str | None = None,
) -> dict:
    """Shared RAG pipeline executing embedding, retrieval, distance filtering,
    Cohere reranking, context building, and prompt formatting.
    Used by both /ask and /ask-stream.
    """
    query_vector = embed_query(question)
    chunks = search_chunks(
        query_vector=query_vector,
        tenant_id=tenant_id,
        source_filter=source_filter,
        limit=RETRIEVAL_TOP_K,
    )
    if not chunks:
        return {
            "has_chunks": False,
            "chunks": [],
            "reranked": [],
            "selected_chunks": [],
            "context": "",
            "prompt": "",
            "sources": [],
        }

    reranked = rerank_chunks(
        question=question,
        chunks=chunks,
        top_k=RERANK_TOP_K,
    )
    selected_chunks = reranked if reranked else chunks[:RERANK_TOP_K]

    context_parts = [
        f"[Document: {chunk.get('source', 'Unknown')} | Page: {chunk.get('page', 'N/A')}]\n{chunk.get('content', '').strip()}"
        for chunk in selected_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""
    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
        }
        for chunk in selected_chunks
    ]

    return {
        "has_chunks": True,
        "chunks": chunks,
        "reranked": reranked,
        "selected_chunks": selected_chunks,
        "context": context,
        "prompt": prompt,
        "sources": sources,
    }


def answer_question(
    question: str,
    tenant_id: str,
    source_filter: str | None = None,
) -> dict:
    """Answers a question using the shared RAG pipeline and shared Gemini client."""
    client = get_gemini_client()

    if is_conversational_or_greeting(question):
        greeting_prompt = get_greeting_prompt(question, tenant_id=tenant_id)
        resp = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=greeting_prompt,
        )
        return {
            "answer": resp.text or "Hello! How can I assist you with your documents today?",
            "sources": [],
        }

    pipeline_result = prepare_rag_pipeline(
        question=question,
        tenant_id=tenant_id,
        source_filter=source_filter,
    )
    if not pipeline_result["has_chunks"]:
        return {
            "answer": "I could not find relevant information in the ingested document.",
            "sources": [],
        }

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=pipeline_result["prompt"],
    )
    return {
        "answer": response.text or "No answer generated",
        "sources": pipeline_result["sources"],
    }
