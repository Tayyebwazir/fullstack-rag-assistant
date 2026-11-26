from app.config import (
    DISTANCE_THRESHOLD,
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


def get_greeting_prompt(question: str) -> str:
    """Shared greeting prompt for both standard and streaming responses."""
    return f"""You are a helpful and polite document assistant.
Respond naturally, politely and concisely to the following greeting or general question:
"{question}"
Remind the user briefly that they can ask questions about their uploaded PDF documents."""


def _check_and_log_benchmark_evaluation(question: str, selected_chunks: list[dict]):
    """If a live query in FastAPI matches a benchmark question, log Expected/Hit Rank/Precision."""
    import inspect
    for frame_info in inspect.stack()[1:6]:
        if "evaluate_rag.py" in frame_info.filename or "test_evaluate_rag.py" in frame_info.filename:
            return  # evaluation scripts format their own output

    try:
        from evaluate_rag import EVAL_SET
    except Exception:
        return

    clean_q = question.strip().lower()
    for item in EVAL_SET:
        eval_q = item["question"].strip().lower()
        if clean_q == eval_q or eval_q in clean_q or clean_q in eval_q:
            qid = item["id"]
            expected = item.get("expected_pages", [])
            is_oos = item.get("out_of_scope", False)
            retrieved_pages = [c.get("page") for c in selected_chunks]

            print(f"\n{'=' * 75}", flush=True)
            if is_oos:
                correctly_rejected = len(selected_chunks) == 0
                status = "PASS" if correctly_rejected else "FAIL"
                action = 'Triggered "I don\'t know"' if correctly_rejected else f"Retrieved {len(selected_chunks)} chunks"
                print(f"[Benchmark Match] [{status}] Q{qid:02d}: {item['question']} [Out-of-Scope]", flush=True)
                print(f"       Expected: None (I don't know) | Retrieved: {retrieved_pages} | Status: {action}", flush=True)
            else:
                first_hit_rank_val = None
                for rank, p in enumerate(retrieved_pages, start=1):
                    if p in expected:
                        first_hit_rank_val = rank
                        break

                first_hit_rank_str = str(first_hit_rank_val) if first_hit_rank_val is not None else "-"
                correct_chunks = sum(1 for p in retrieved_pages if p in expected)
                total_chunks = len(selected_chunks)
                precision_k = (correct_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
                status = "PASS" if first_hit_rank_val is not None else "FAIL"

                print(f"[Benchmark Match] [{status}] Q{qid:02d}: {item['question']}", flush=True)
                print(f"       Expected: {expected} | Retrieved: {retrieved_pages} | Hit Rank: {first_hit_rank_str} | Precision@{total_chunks}: {precision_k:.1f}% ({correct_chunks}/{total_chunks})", flush=True)
            print(f"{'=' * 75}\n", flush=True)
            break


def prepare_rag_pipeline(
    question: str,
    tenant_id: str,
    source_filter: str | None = None,
    use_rerank: bool = True,
    top_k: int = RERANK_TOP_K,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """Shared RAG pipeline executing embedding, retrieval (20 chunks), distance filtering,
    optional Cohere reranking (top 4 chunks), context building, and prompt formatting.
    Used by /ask, /ask-stream, and evaluation workflows.
    """
    query_vector = embed_query(question)
    chunks = search_chunks(
        query_vector=query_vector,
        tenant_id=tenant_id,
        source_filter=source_filter,
        limit=RETRIEVAL_TOP_K,
        distance_threshold=distance_threshold,
    )
    if not chunks:
        _check_and_log_benchmark_evaluation(question, [])
        return {
            "has_chunks": False,
            "chunks": [],
            "reranked": [],
            "selected_chunks": [],
            "context": "",
            "prompt": "",
            "sources": [],
        }

    reranked = []
    if use_rerank:
        reranked = rerank_chunks(
            question=question,
            chunks=chunks,
            top_k=top_k,
        )
    selected_chunks = reranked if reranked else chunks[:top_k]
    _check_and_log_benchmark_evaluation(question, selected_chunks)

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
        greeting_prompt = get_greeting_prompt(question)
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


def stream_answer_question(
    question: str,
    tenant_id: str,
    source_filter: str | None = None,
):
    """Streams RAG responses (events) using the shared RAG pipeline."""
    client = get_gemini_client()

    if is_conversational_or_greeting(question):
        greeting_prompt = get_greeting_prompt(question)
        stream = client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=greeting_prompt,
        )
        yield {"type": "metadata", "chunks": [], "reranked": []}
        for chunk in stream:
            if chunk.text:
                yield {"type": "token", "token": chunk.text}
        yield {"type": "done"}
        return

    pipeline_result = prepare_rag_pipeline(
        question=question,
        tenant_id=tenant_id,
        source_filter=source_filter,
    )

    yield {
        "type": "metadata",
        "chunks": pipeline_result["chunks"],
        "reranked": pipeline_result["reranked"],
    }

    if not pipeline_result["has_chunks"]:
        yield {
            "type": "token",
            "token": "I could not find relevant information in the ingested document.",
        }
        yield {"type": "done"}
        return

    stream = client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=pipeline_result["prompt"],
    )
    for chunk in stream:
        if chunk.text:
            yield {"type": "token", "token": chunk.text}

    yield {"type": "done"}
