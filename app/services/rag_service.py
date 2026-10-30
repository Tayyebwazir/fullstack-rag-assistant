from google import genai
from app.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    TOP_K,
)
from app.services.embeddings import embed_query
from app.services.vector_store import search_chunks
from app.services.intent import is_conversational_or_greeting

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def answer_question(
    question: str,
    tenant_id: str = "default_user",
    source_filter: str | None = None,
) -> dict:
    # Handle greetings / general conversation directly without RAG
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
        limit=TOP_K,
    )
    if not chunks:
        return {
            "answer": "I could not find relevant information in the ingested document.",
            "sources": [],
        }

    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[source:{chunk['source']},page:{chunk['page']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)
    prompt = f"""You are a document question answering assistant.
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
            for chunk in chunks
        ],
    }
