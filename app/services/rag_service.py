from google import genai
from app.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from app.services.embeddings import embed_query
from app.services.vector_store import search_chunks
from app.services.intent import is_conversational_or_greeting
from app.services.reranker import rerank_chunks

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
SYSTEM_PROMPT = """You are a helpful student AI assistant.

Rules:
1. Answer questions ONLY using the provided context.
2. Do not use outside knowledge.
3. If the context does not contain the answer, say: "I don't know based on the provided documents."
4. Always cite the source document and page number.
5. Do not invent information or citations.
"""

def answer_question(
    question: str,
    tenant_id: str = "alpha",
    source_filter: str | None = None,
) -> dict:
    
    if is_conversational_or_greeting(question):
        greeting_prompt = f"""You are a helpful and polite document assistant.
Respond naturally, politely and concisely to the following greeting or general question:
"{question}"
Remind the user briefly that they can ask questions about their uploaded PDF documents."""
        resp = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=greeting_prompt,
        )
        return {
            "answer": resp.text or "Hello! How can I assist you with your documents today?",
            "sources": [],
        }

    query_vector = embed_query(question)
    chunks = search_chunks(
        query_vector=query_vector,
        tenant_id=tenant_id,
        source_filter=source_filter,
        limit=RETRIEVAL_TOP_K,
    )
    if not chunks:
        return {
            "answer": "I could not find relevant information in the ingested document.",
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

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return {
        "answer": response.text or "No answer generated",
        "sources": [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
            }
            for chunk in selected_chunks
        ],
    }
