import re

COMMON_CONVERSATIONAL = {
    "hello",
    "hi",
    "hey",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what are you",
    "how are you",
    "what can you do",
    "help",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
}


def is_conversational_or_greeting(text: str) -> bool:
    """Detects simple greetings or chit-chat that don't need RAG retrieval."""
    clean = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not clean:
        return True
    if clean in COMMON_CONVERSATIONAL:
        return True
    for prefix in ["hello", "hi", "hey"]:
        if clean.startswith(f"{prefix} ") or clean == prefix:
            if len(clean.split()) <= 4:
                return True
    return False

