import json
import logging
from google import genai
from google.genai import types
from app.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    RERANK_TOP_K,
)

logger = logging.getLogger(__name__)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    if not chunks:
        return []

    items = [f"[{i}] {c.get('content', '')[:600]}" for i, c in enumerate(chunks)]
    prompt = f"""You are a retrieval reranker. Score each document chunk's relevance to the question from 0.0 to 10.0 (10.0 = exact answer, 0.0 = completely irrelevant).
Question: {question}

Candidate Chunks:
{chr(10).join(items)}

Return ONLY a JSON array of objects with integer 'index' and float 'score', e.g.:
[{{"index": 0, "score": 9.5}}, {{"index": 1, "score": 2.0}}]"""

    score_map = {}
    try:
        response = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = json.loads(response.text or "[]")
        score_map = {int(item["index"]): float(item.get("score", 0.0)) for item in data if "index" in item}
    except Exception as exc:
        logger.error("Gemini reranker failed (%s); keeping vector search order.", exc)

    reranked = [
        {
            **chunk,
            "original_rank": orig + 1,
            "relevance_score": score_map.get(orig, 0.0),
        }
        for orig, chunk in enumerate(chunks)
    ]

    # Only re-sort if scoring succeeded
    if score_map:
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)

    for new_rank, item in enumerate(reranked, start=1):
        item["rerank_position"] = new_rank

    return reranked[:top_k] if top_k else reranked