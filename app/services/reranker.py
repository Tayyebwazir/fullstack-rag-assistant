import logging
import time
import requests
from typing import Any
from app.config import COHERE_API_KEY, RERANK_MODEL, RERANK_TOP_K

logger = logging.getLogger(__name__)


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """Reranks candidate chunks using Cohere Rerank API.

    Args:
        question: User query string.
        chunks: List of candidate chunks from Weaviate (~20 candidates).
        top_k: Number of chunks to select after reranking (e.g. 4).

    Returns:
        Top-k chunks ordered by relevance score descending.
    """
    if not chunks:
        return []

    if not COHERE_API_KEY or COHERE_API_KEY.startswith("your-"):
        import warnings
        msg = (
            "COHERE_API_KEY is missing or unconfigured in .env. "
            f"Cohere reranking cannot execute; falling back to top {top_k} vector search candidates."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
        print(f"\n[WARNING] {msg}\n")
        return chunks[:top_k]

    try:
        # 1. Extract chunk texts
        documents = [c.get("content", "") for c in chunks]

        # 2. Call Cohere Rerank v2 API with rate-limit retry
        response = None
        for attempt in range(4):
            response = requests.post(
                "https://api.cohere.com/v2/rerank",
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": RERANK_MODEL,
                    "query": question,
                    "documents": documents,
                    "top_n": top_k,
                },
                timeout=12,
            )

            if response.status_code == 429 and attempt < 3:
                wait_sec = 6 * (attempt + 1)
                time.sleep(wait_sec)
                continue
            break

        if response is None or response.status_code != 200:
            err_msg = response.text if response is not None else "No response"
            logger.error("Cohere Rerank API returned error: %s", err_msg)
            return chunks[:top_k]

        data = response.json()
        results = data.get("results", [])

        # 3. Rebuild chunk list with new ranks and scores
        reranked = []
        for new_rank, item in enumerate(results, start=1):
            orig_idx = item["index"]
            chunk = chunks[orig_idx].copy()
            chunk["original_rank"] = orig_idx + 1
            chunk["relevance_score"] = float(item["relevance_score"])
            chunk["rerank_position"] = new_rank
            reranked.append(chunk)

        # 4. Log reranking results to server terminal
        print(f"\n{'=' * 65}")
        print(f"[Reranker] Query: '{question}'")
        print(f"[Reranker] Evaluated {len(chunks)} candidates -> Selected Top {len(reranked)}:")
        for c in reranked:
            rank = c["rerank_position"]
            orig = c["original_rank"]
            score = c["relevance_score"]
            page = c.get("page", "N/A")
            preview = c.get("content", "")[:500].replace("\n", " ")
            print(f"  Rank #{rank} (Was #{orig}) | Score: {score:8.4f} | Page {str(page):2} | {preview}...")
        print(f"{'=' * 65}\n")

        return reranked

    except Exception as exc:
        logger.error("Cohere Rerank execution failed: %s; keeping vector search order.", exc)
        return chunks[:top_k]
