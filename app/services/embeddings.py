import time
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("Set gemini_api_key in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Ordered models to fallback on if one model's daily quota is exhausted (429)
def _get_candidate_models():
    primary = EMBEDDING_MODEL or "gemini-embedding-2"
    fallbacks = ["gemini-embedding-2", "gemini-embedding-2-preview", "gemini-embedding-001"]
    ordered = [primary] + [m for m in fallbacks if m != primary]
    return ordered


def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    if not texts:
        return []

    models = _get_candidate_models()
    last_err = None

    for model_name in models:
        try:
            all_vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                for attempt in range(3):
                    try:
                        response = gemini_client.models.embed_content(
                            model=model_name,
                            contents=batch,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                        )
                        vectors = [list(e.values) for e in response.embeddings]
                        all_vectors.extend(vectors)
                        time.sleep(0.3)
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str and attempt < 2:
                            time.sleep(5 * (attempt + 1))
                        else:
                            raise e
            return all_vectors
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                print(f"[Embeddings] Model '{model_name}' hit 429 quota limit. Trying next candidate model...")
                last_err = e
                continue
            raise e

    if last_err:
        raise last_err
    return []


def embed_document(text: str) -> list[float]:
    return embed_documents([text])[0]


def embed_query(question: str) -> list[float]:
    models = _get_candidate_models()
    last_err = None

    for model_name in models:
        try:
            response = gemini_client.models.embed_content(
                model=model_name,
                contents=question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return list(response.embeddings[0].values)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                print(f"[Embeddings] Model '{model_name}' quota exhausted (429). Falling back to next model...")
                last_err = e
                continue
            raise e

    if last_err:
        raise last_err
    return []
