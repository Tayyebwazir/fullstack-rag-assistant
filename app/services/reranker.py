import json 
from google import genai
from app.config import(
    GEMINI_API_KEY,
    GENERATION_MODEL,
    TOP_K,
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    if not chunks:
        return []
    items = [f"[{i}] {c.get('content', '')[:400]}" for i, c in enumerate(chunks)]
    prompt = f"""Score each chunk's relevance to the question from 0.0 to 10.0.
Output ONLY a JSON array: [{{"index": 0, "score": 9.5}}, ...]
Question: {question}
Chunks:
{chr(10).join(items)}
"""
    try:
        response = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        text = (response.text or "").replace("```json", "").replace("```", "").strip()
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            data = json.loads(text[start:end+1])
            score_map = {int(item["index"]): float(item.get("score", 0.0)) for item in data if "index" in item}
        else:
            score_map = {}
    except Exception:
        score_map = {}
    reranked = [
        {
            **chunk,
            "original_rank": orig + 1,
            "relevance_score": score_map.get(orig, 1.0),
        }
        for orig, chunk in enumerate(chunks)
    ]
    reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    for new_rank, item in enumerate(reranked, start=1):
        item["rerank_position"] = new_rank
    return reranked[:top_k] if top_k else reranked
