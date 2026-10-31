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

    context_parts = [chunk["content"].strip() for chunk in chunks]
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are an expert AI document assistant.
Provide a clear, detailed, comprehensive, and well-structured answer to the user's question based strictly on the provided context.

Context:
{context}

Question:
{question}

Guidelines:
- Provide high quality, well-organized explanations with Markdown formatting (use headings, bullet points, bold keywords, and code snippets when present).
- Do NOT output citation brackets or source tags (such as [source:...], [page:...]) inside your answer text. All document sources and page locations are already presented in a dedicated visual panel.
- Explain technical definitions and provide code examples if available in the text.
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
