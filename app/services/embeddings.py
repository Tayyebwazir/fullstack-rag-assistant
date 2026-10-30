import time
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("Set gemini_api_key in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    if not texts:
        return []

    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                response = gemini_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                vectors = [list(e.values) for e in response.embeddings]
                all_vectors.extend(vectors)
                time.sleep(0.5)
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "resource_exhausted" in err_str) and attempt < 4:
                    wait_time = 16 * (attempt + 1)
                    print(f"[Embeddings] Rate limit 429 hit. Waiting {wait_time}s before retry (attempt {attempt + 1}/5)...")
                    time.sleep(wait_time)
                else:
                    raise e

    return all_vectors


def embed_document(text: str) -> list[float]:
    return embed_documents([text])[0]


def embed_query(question: str) -> list[float]:
    for attempt in range(3):
        try:
            response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return list(response.embeddings[0].values)
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "resource_exhausted" in err_str) and attempt < 2:
                time.sleep(10)
                continue
            raise e
