import time
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("Set gemini_api_key in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory query vector cache to prevent repeated rate limits
_query_cache: dict[str, list[float]] = {}


def _get_candidate_models():
    # If one model's quota is exhausted (429), fall back across available 3072-dim models
    candidates = ["gemini-embedding-001", "gemini-embedding-2", "gemini-embedding-2-preview"]
    configured = (EMBEDDING_MODEL or "").strip()
    if configured and configured in candidates:
        ordered = [configured] + [m for m in candidates if m != configured]
        return ordered
    return candidates


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
                contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
                for attempt in range(5):
                    try:
                        response = gemini_client.models.embed_content(
                            model=model_name,
                            contents=contents,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                        )
                        vectors = [list(e.values) for e in response.embeddings]
                        all_vectors.extend(vectors)
                        time.sleep(0.3)
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str and attempt < 4:
                            wait_sec = 18 * (attempt + 1)
                            print(f"[Embeddings] Rate limit (RPM) hit on {model_name}. Waiting {wait_sec}s for quota window reset...")
                            time.sleep(wait_sec)
                        else:
                            raise e
            if len(all_vectors) == len(texts):
                return all_vectors
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                print(f"[Embeddings] Model '{model_name}' rate-limited. Trying fallback model...")
                last_err = e
                continue
            raise e

    if last_err:
        raise last_err
    return []


def embed_document(text: str) -> list[float]:
    return embed_documents([text])[0]


def embed_query(question: str) -> list[float]:
    norm_q = question.strip().lower()
    if norm_q in _query_cache:
        return _query_cache[norm_q]

    models = _get_candidate_models()
    last_err = None

    for model_name in models:
        for attempt in range(3):
            try:
                response = gemini_client.models.embed_content(
                    model=model_name,
                    contents=question,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                vec = list(response.embeddings[0].values)
                _query_cache[norm_q] = vec
                return vec
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str:
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    print(f"[Embeddings] Model '{model_name}' exhausted. Trying next model...")
                    last_err = e
                    break
                raise e

    if last_err:
        raise last_err
    return []
