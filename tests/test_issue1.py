"""Test Issue 1: Fix /ask Greeting Crash.

Verifies:
1. Authenticated user sending "hi" to /ask returns HTTP 200 with a valid greeting response.
2. Unauthenticated user sending "hi" to /ask returns HTTP 401.
3. Authenticated user attempting cross-tenant access returns HTTP 403.
4. /ask-stream continues to work for greetings with SSE events.
5. get_greeting_prompt(question) works without unexpected arguments.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_user
from app.services.rag_service import get_greeting_prompt, is_conversational_or_greeting


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleaned after each test."""
    yield
    app.dependency_overrides.clear()


def test_get_greeting_prompt_signature():
    """Verify get_greeting_prompt accepts only question and produces valid prompt."""
    prompt = get_greeting_prompt("hi")
    assert isinstance(prompt, str)
    assert "hi" in prompt
    assert is_conversational_or_greeting("hi") is True


def test_ask_greeting_authenticated_success():
    """Verify /ask returns 200 and a valid greeting when authenticated user says 'hi'."""
    mock_user = {"uid": "user_issue1", "email": "issue1@example.com"}
    mock_record = {
        "uid": "user_issue1",
        "email": "issue1@example.com",
        "tenant_id": "tenant_issue1",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = "Hello! How can I assist you with your documents today?"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_resp

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_client):

        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "hi"})

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert "answer" in data
            assert len(data["answer"]) > 0
            assert "Hello" in data["answer"]
            assert data["sources"] == []
            mock_client.models.generate_content.assert_called_once()


def test_ask_greeting_unauthenticated_returns_401():
    """Verify /ask rejects unauthenticated requests with 401."""
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"


def test_ask_greeting_cross_tenant_returns_403():
    """Verify tenant isolation: requesting a foreign tenant returns 403."""
    mock_user = {"uid": "user_issue1", "email": "issue1@example.com"}
    mock_record = {
        "uid": "user_issue1",
        "email": "issue1@example.com",
        "tenant_id": "tenant_issue1",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)):

        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "hi", "tenant_id": "unauthorized_tenant_xyz"})
            assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
            assert "Access Denied" in response.json().get("detail", "")


def test_ask_stream_greeting_remains_unaffected():
    """Verify /ask-stream continues to stream greetings properly."""
    mock_user = {"uid": "user_issue1", "email": "issue1@example.com"}
    mock_record = {
        "uid": "user_issue1",
        "email": "issue1@example.com",
        "tenant_id": "tenant_issue1",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_chunk = MagicMock()
    mock_chunk.text = "Hello! How can I help?"

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [mock_chunk]

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_issue1", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_client):

        with TestClient(app) as client:
            response = client.post("/ask-stream", json={"question": "hello"})
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            content = response.text
            assert "data:" in content
            assert "done" in content
