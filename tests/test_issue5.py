"""Test Issue 5: Add Comprehensive Automated Tests.

Covers:
1. Authentication: Verify all protected API endpoints reject missing and invalid authentication tokens with HTTP 401.
2. CORS: Verify configured origins are allowed and unauthorized origins are blocked.
3. Intent Router: Verify greetings, knowledge questions, and mixed inputs (e.g. 'hey, what is attention?') are classified correctly.
4. Tenant Security: Verify missing, empty, invalid, and unauthorized tenant IDs are rejected; cross-tenant access is denied; no fallback tenant is used.
5. Greeting Regression: Verify sending 'hi' to /ask as an authenticated user returns HTTP 200 and a valid greeting instead of HTTP 500.
6. Streaming: Verify /ask-stream continues working correctly for both greetings and document queries.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import ALLOWED_ORIGINS
from app.services.auth import get_current_user
from app.services.intent import is_conversational_or_greeting


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are reset after every test."""
    yield
    app.dependency_overrides.clear()


# ==============================================================================
# 1. AUTHENTICATION TESTS
# ==============================================================================

PROTECTED_ENDPOINTS = [
    ("POST", "/ask", {"json": {"question": "What is attention?"}}),
    ("POST", "/question/ask", {"json": {"question": "What is attention?"}}),
    ("POST", "/ask-stream", {"json": {"question": "What is attention?"}}),
    ("GET", "/documents/list", {}),
    ("GET", "/documents/sources", {}),
    ("GET", "/sources", {}),
    ("POST", "/documents/ingest", {"json": {"doc_id": "test_doc"}}),
    ("POST", "/ingest", {"json": {"doc_id": "test_doc"}}),
    ("DELETE", "/documents/delete", {"json": {"doc_id": "test_doc"}}),
    ("GET", "/documents/test_doc/extracted-text", {}),
    ("GET", "/user/profile", {}),
]


@pytest.mark.parametrize("method,endpoint,kwargs", PROTECTED_ENDPOINTS)
def test_protected_endpoints_reject_missing_token_with_401(method: str, endpoint: str, kwargs: dict):
    """Verify that every protected endpoint rejects requests lacking authentication with HTTP 401."""
    with TestClient(app) as client:
        if method == "GET":
            res = client.get(endpoint, **kwargs)
        elif method == "POST":
            res = client.post(endpoint, **kwargs)
        elif method == "DELETE":
            res = client.request("DELETE", endpoint, **kwargs)
        else:
            pytest.fail(f"Unsupported method: {method}")

        assert res.status_code == 401, (
            f"Endpoint {method} {endpoint} expected 401 on missing auth, got {res.status_code}: {res.text}"
        )
        assert "WWW-Authenticate" in res.headers


@pytest.mark.parametrize("method,endpoint,kwargs", PROTECTED_ENDPOINTS)
def test_protected_endpoints_reject_invalid_token_with_401(method: str, endpoint: str, kwargs: dict):
    """Verify that every protected endpoint rejects requests with invalid/expired tokens with HTTP 401."""
    invalid_headers = {"Authorization": "Bearer invalid.untrusted.token"}
    call_kwargs = dict(kwargs)
    call_kwargs["headers"] = invalid_headers

    with patch("app.services.auth.FIREBASE_PROJECT_ID", "mock-rag-project"), \
         patch("app.services.auth.id_token.verify_firebase_token", side_effect=ValueError("Token signature expired or invalid")):

        with TestClient(app) as client:
            if method == "GET":
                res = client.get(endpoint, **call_kwargs)
            elif method == "POST":
                res = client.post(endpoint, **call_kwargs)
            elif method == "DELETE":
                res = client.request("DELETE", endpoint, **call_kwargs)
            else:
                pytest.fail(f"Unsupported method: {method}")

            assert res.status_code == 401, (
                f"Endpoint {method} {endpoint} expected 401 on invalid auth, got {res.status_code}: {res.text}"
            )


# ==============================================================================
# 2. CORS TESTS
# ==============================================================================

def test_cors_configured_origins_allowed():
    """Verify preflight and actual requests from configured origins are allowed."""
    with TestClient(app) as client:
        for origin in ALLOWED_ORIGINS:
            # Preflight OPTIONS request
            res = client.options(
                "/ask",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization,Content-Type",
                },
            )
            assert res.status_code == 200, f"Expected 200 for OPTIONS from {origin}, got {res.status_code}"
            assert res.headers.get("access-control-allow-origin") == origin
            assert res.headers.get("access-control-allow-credentials") == "true"


def test_cors_unauthorized_origins_blocked():
    """Verify requests and preflights from unauthorized origins do not receive CORS allow headers."""
    unauthorized_origin = "http://malicious-attacker.com"
    with TestClient(app) as client:
        res = client.options(
            "/ask",
            headers={
                "Origin": unauthorized_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORSMiddleware does not return Access-Control-Allow-Origin for untrusted origins
        assert res.headers.get("access-control-allow-origin") != unauthorized_origin


# ==============================================================================
# 3. INTENT ROUTER TESTS
# ==============================================================================

def test_intent_router_pure_greetings():
    """Verify standalone greetings and conversational chit-chat are classified as greetings."""
    greetings = [
        "hello",
        "hi",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "yo",
        "who are you",
        "what can you do",
        "help",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
    ]
    for text in greetings:
        assert is_conversational_or_greeting(text) is True, f"Failed for greeting: '{text}'"


def test_intent_router_knowledge_questions():
    """Verify actual questions and document inquiries are classified as knowledge queries (not greetings)."""
    knowledge_queries = [
        "what is the transformer architecture?",
        "how does self-attention work?",
        "explain multi-head attention",
        "what are the positional encoding formulas?",
        "what dataset was used for training?",
        "how many layers are in the encoder?",
        "calculate the hit@1 and MRR metrics",
    ]
    for text in knowledge_queries:
        assert is_conversational_or_greeting(text) is False, f"Expected False for knowledge query: '{text}'"


def test_intent_router_mixed_inputs_prevent_retrieval_skip():
    """Verify mixed inputs starting with a greeting prefix still route to document retrieval."""
    mixed_queries = [
        "hey, what is attention?",
        "hi attention is all you need",
        "hello, can you explain the encoder layers?",
        "hey what is attention",
        "hi how does the model work",
        "hello who is the author of this paper",
        "hi can you tell me about the BLEU score",
    ]
    for text in mixed_queries:
        assert is_conversational_or_greeting(text) is False, (
            f"Mixed query '{text}' incorrectly classified as greeting; document retrieval would be skipped!"
        )


# ==============================================================================
# 4. TENANT SECURITY TESTS
# ==============================================================================

def test_tenant_security_missing_or_empty_tenant_rejected_with_403():
    """Verify users with missing or empty tenant IDs cannot access protected routes."""
    mock_user = {"uid": "user_empty_tenant", "email": "empty@example.com"}
    mock_record = {
        "uid": "user_empty_tenant",
        "email": "empty@example.com",
        "tenant_id": "",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.get_user_record", return_value=mock_record), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "", "Access Denied: No tenant assigned.")):

        with TestClient(app) as client:
            res = client.post("/ask", json={"question": "What is the Transformer architecture?"})
            assert res.status_code == 403
            assert "Access Denied" in res.json().get("detail", "")


def test_tenant_security_banned_fallback_tenants_rejected():
    """Verify that banned fallback values ('default', 'alpha', 'beta', 'tri') are rejected without fallbacks."""
    banned_tenants = ["default", "alpha", "beta", "tri", "tenant_xyz"]
    mock_user = {"uid": "user_banned_tenant", "email": "banned@example.com"}

    app.dependency_overrides[get_current_user] = lambda: mock_user

    for banned in banned_tenants:
        mock_record = {
            "uid": "user_banned_tenant",
            "email": "banned@example.com",
            "tenant_id": banned,
            "permissions": ["READ"],
            "is_active": True,
        }
        with patch("app.dependencies.get_user_record", return_value=mock_record), \
             patch("app.dependencies.verify_tenant_access", return_value=(False, f"Access Denied: Fallback or invalid tenant '{banned}' is prohibited.")):

            with TestClient(app) as client:
                res = client.post("/ask", json={"question": "What is attention?"})
                assert res.status_code == 403, f"Expected 403 for banned tenant '{banned}', got {res.status_code}"


def test_tenant_security_cross_tenant_access_denied():
    """Verify an authenticated user assigned to tenant_A is forbidden from requesting tenant_B."""
    mock_user = {"uid": "user_a", "email": "a@example.com"}
    mock_record = {
        "uid": "user_a",
        "email": "a@example.com",
        "tenant_id": "tenant_a",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_a", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_a", None)):

        with TestClient(app) as client:
            # Attempt to query tenant_b
            res = client.post("/ask", json={"question": "What is attention?", "tenant_id": "tenant_b"})
            assert res.status_code == 403
            assert "Access Denied" in res.json().get("detail", "")
            assert "tenant_b" in res.json().get("detail", "")


def test_tenant_security_cross_tenant_document_access_denied():
    """Verify an authenticated user cannot access or read documents belonging to another tenant."""
    mock_user = {"uid": "user_a", "email": "a@example.com"}
    mock_record = {
        "uid": "user_a",
        "email": "a@example.com",
        "tenant_id": "tenant_a",
        "permissions": ["READ"],
        "is_active": True,
    }

    # Document belongs to tenant_b
    foreign_doc = {
        "doc_id": "doc_foreign",
        "tenant_id": "tenant_b",
        "filename": "confidential.pdf",
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_a", None)), \
         patch("app.dependencies.get_document_by_id", return_value=foreign_doc):

        with TestClient(app) as client:
            res = client.get("/documents/doc_foreign/extracted-text")
            assert res.status_code == 403
            assert "Access Denied" in res.json().get("detail", "")


# ==============================================================================
# 5. GREETING REGRESSION TESTS
# ==============================================================================

def test_greeting_regression_authenticated_hi_returns_200():
    """Verify sending 'hi' to /ask returns HTTP 200 with a valid greeting and does NOT return HTTP 500."""
    mock_user = {"uid": "user_greeting_test", "email": "greeting@example.com"}
    mock_record = {
        "uid": "user_greeting_test",
        "email": "greeting@example.com",
        "tenant_id": "tenant_greeting",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = "Hello! I am your AI assistant for your uploaded documents."

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.return_value = mock_gemini_resp

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_greeting", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_greeting", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_gemini_client):

        with TestClient(app) as client:
            res = client.post("/ask", json={"question": "hi"})
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert "answer" in data
            assert len(data["answer"]) > 0
            assert data.get("sources") == []


# ==============================================================================
# 6. STREAMING TESTS
# ==============================================================================

def test_streaming_greeting_works_correctly():
    """Verify /ask-stream streams events for greetings without invoking vector search."""
    mock_user = {"uid": "user_stream_greet", "email": "stream_greet@example.com"}
    mock_record = {
        "uid": "user_stream_greet",
        "email": "stream_greet@example.com",
        "tenant_id": "tenant_stream",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_chunk = MagicMock()
    mock_chunk.text = "Hello! How can I help you today?"

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content_stream.return_value = [mock_chunk]

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_stream", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_stream", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_gemini_client):

        with TestClient(app) as client:
            res = client.post("/ask-stream", json={"question": "hello"})
            assert res.status_code == 200
            assert "text/event-stream" in res.headers.get("content-type", "")
            events = res.text
            assert "metadata" in events
            assert "token" in events
            assert "done" in events


def test_streaming_rag_query_works_correctly():
    """Verify /ask-stream streams metadata, tokens, and done events for document queries."""
    mock_user = {"uid": "user_stream_rag", "email": "stream_rag@example.com"}
    mock_record = {
        "uid": "user_stream_rag",
        "email": "stream_rag@example.com",
        "tenant_id": "tenant_stream",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_token1 = MagicMock()
    mock_token1.text = "The Transformer is "
    mock_token2 = MagicMock()
    mock_token2.text = "an attention-based architecture."

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content_stream.return_value = [mock_token1, mock_token2]

    retrieved_chunks = [
        {"source": "attention.pdf", "page": 1, "chunk_index": 0, "content": "Transformer architecture."}
    ]

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_stream", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_stream", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_gemini_client), \
         patch("app.services.rag_service.embed_query", return_value=[0.05] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=retrieved_chunks), \
         patch("app.services.rag_service.rerank_chunks", return_value=retrieved_chunks):

        with TestClient(app) as client:
            res = client.post("/ask-stream", json={"question": "What is the Transformer?"})
            assert res.status_code == 200
            assert "text/event-stream" in res.headers.get("content-type", "")
            events = res.text
            assert "metadata" in events
            assert "attention.pdf" in events
            assert "token" in events
            assert "done" in events
