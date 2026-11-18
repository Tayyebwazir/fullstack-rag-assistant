import re

COMMON_CONVERSATIONAL = {
    "hello",
    "hi",
    "hey",
    "hi there",
    "hello there",
    "hey there",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what are you",
    "how are you",
    "how are you doing",
    "what can you do",
    "help",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "howdy",
    "yo",
}

GREETING_FILLERS = {"there", "bot", "assistant", "ai", "friend", "team"}


def is_conversational_or_greeting(text: str) -> bool:
    """Detects simple greetings or chit-chat that don't need RAG retrieval.

    Ensures that queries with actual questions or document topics (e.g.
    'hey what is attention', 'hi attention is all you need', 'hey attention mechanism')
    never skip document retrieval.
    """
    clean = re.sub(r"[^\w\s]", " ", text.strip().lower())
    clean = " ".join(clean.split())

    if not clean:
        return True

    # 1. Exact match with common conversational phrases
    if clean in COMMON_CONVERSATIONAL:
        return True

    # 2. Check for greeting prefix (hi, hello, hey)
    words = clean.split()
    if words and words[0] in {"hello", "hi", "hey"}:
        remaining_words = words[1:]

        # Only treat as greeting if nothing remains or only filler words
        if not remaining_words or all(
            w in GREETING_FILLERS for w in remaining_words
        ):
            return True

    return False