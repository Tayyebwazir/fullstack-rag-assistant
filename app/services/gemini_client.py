"""Reusable Google Gemini client singleton per application process."""

import logging
from google import genai
from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """Returns the process-wide reusable Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not configured in environment variables.")
            raise RuntimeError("Set GEMINI_API_KEY in .env")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client
