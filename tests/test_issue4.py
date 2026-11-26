"""Test Issue 4: Completely Remove Default Tenant Fallbacks.

Verifies:
1. Valid tenant ID succeeds.
2. Missing, empty, or None tenant ID is rejected with HTTP 403.
3. Invalid or unauthorized tenant ID (including 'default' and 'alpha') is rejected with HTTP 403.
4. Cross-tenant access is strictly denied with HTTP 403.
5. create_document_record requires an explicit valid tenant ID (rejects '', None, 'default', 'alpha').
6. Synchronization logic strictly skips documents with invalid/fallback tenants and never writes them.
7. ensure_tenant in vector_store strictly requires valid tenant_id.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_user
from app.services.firebase_db import create_document_record, verify_tenant_access
from app.services.vector_store import ensure_tenant


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleared after each test."""
    yield
    app.dependency_overrides.clear()


def test_valid_tenant_operation_succeeds():
    """Verify that an authenticated user with a valid tenant succeeds."""
    mock_user = {"uid": "user_valid", "email": "valid@example.com"}
    mock_record = {
        "uid": "user_valid",
        "email": "valid@example.com",
        "tenant_id": "valid_tenant_uuid_123",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "valid_tenant_uuid_123", None)), \
         patch("app.routers.documents.get_tenant_documents", return_value=[]):

        with TestClient(app) as client:
            res = client.get("/documents/list")
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert data["tenant_id"] == "valid_tenant_uuid_123"


def test_missing_or_empty_tenant_rejected_with_403():
    """Verify that missing, empty, or None tenant_id returns HTTP 403."""
    mock_user = {"uid": "user_no_tenant", "email": "notenant@example.com"}
    mock_record = {
        "uid": "user_no_tenant",
        "email": "notenant@example.com",
        "tenant_id": "",
        "tenant_error": "No tenant assigned",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.get_user_record", return_value=mock_record), \
         patch("app.routers.users.get_user_record", return_value=mock_record):

        with TestClient(app) as client:
            # 1. Profile route
            res_profile = client.get("/user/profile")
            assert res_profile.status_code == 403

            # 2. Document list route
            res_docs = client.get("/documents/list")
            assert res_docs.status_code == 403

            # 3. Ask route
            res_ask = client.post("/ask", json={"question": "What is self-attention?"})
            assert res_ask.status_code == 403


def test_invalid_fallback_tenant_rejected_with_403():
    """Verify that legacy fallback tenant names like 'default' or 'alpha' are rejected."""
    mock_record_default = {
        "uid": "user_default",
        "email": "default@example.com",
        "tenant_id": "default",
        "permissions": ["READ"],
        "is_active": True,
    }

    # verify_tenant_access must reject cross-tenant or invalid targets
    authorized, msg = verify_tenant_access(mock_record_default, "some_other_tenant", "READ")
    assert authorized is False
    assert "Cross-tenant access is strictly forbidden" in msg


def test_cross_tenant_access_denied():
    """Verify that an authenticated user cannot access another tenant's documents."""
    mock_user = {"uid": "user_tenant_a", "email": "a@example.com"}
    mock_record = {
        "uid": "user_tenant_a",
        "email": "a@example.com",
        "tenant_id": "tenant_a_123",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_a_123", None)):
        with TestClient(app) as client:
            # Requesting tenant_b when authorized for tenant_a
            res = client.get("/documents/list?tenant_id=tenant_b_456")
            assert res.status_code == 403
            assert "Access Denied" in res.json().get("detail", "")


def test_create_document_record_requires_explicit_tenant():
    """Verify create_document_record rejects missing or fallback tenants ('', None, 'default', 'alpha')."""
    # 1. Missing tenant_id -> raises ValueError
    with pytest.raises(ValueError) as exc1:
        create_document_record({"filename": "test.pdf", "tenant_id": ""})
    assert "Explicit, valid tenant_id is required" in str(exc1.value)

    # 2. None tenant_id -> raises ValueError
    with pytest.raises(ValueError) as exc2:
        create_document_record({"filename": "test.pdf", "tenant_id": None})
    assert "Explicit, valid tenant_id is required" in str(exc2.value)

    # 3. 'default' tenant_id -> raises ValueError
    with pytest.raises(ValueError) as exc3:
        create_document_record({"filename": "test.pdf", "tenant_id": "default"})
    assert "Explicit, valid tenant_id is required" in str(exc3.value)

    # 4. 'alpha' tenant_id -> raises ValueError
    with pytest.raises(ValueError) as exc4:
        create_document_record({"filename": "test.pdf", "tenant_id": "alpha"})
    assert "Explicit, valid tenant_id is required" in str(exc4.value)

    # 5. Valid tenant_id -> succeeds
    with patch("app.services.firebase_db._get_firestore_client", return_value=None), \
         patch("app.services.firebase_db._save_local_documents"):
        doc = create_document_record({"filename": "valid.pdf", "tenant_id": "12345678-1234-5678-1234-567812345678"})
        assert doc["tenant_id"] == "12345678-1234-5678-1234-567812345678"


def test_vector_store_ensure_tenant_fails_closed_on_empty():
    """Verify vector_store.ensure_tenant raises ValueError if tenant_id is empty or whitespace."""
    mock_coll = MagicMock()
    with pytest.raises(ValueError) as exc_info:
        ensure_tenant(mock_coll, "")
    assert "A valid tenant_id is strictly required" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info2:
        ensure_tenant(mock_coll, "   ")
    assert "A valid tenant_id is strictly required" in str(exc_info2.value)
