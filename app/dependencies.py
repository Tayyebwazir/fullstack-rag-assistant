"""Shared authentication, tenant isolation, and document ownership dependencies."""

from fastapi import Depends, HTTPException

from app.services.auth import get_current_user
from app.services.firebase_db import (
    get_document_by_id,
    get_user_record,
    verify_tenant_access,
)


def resolve_authorized_user(
    user: dict,
    required_permission: str | None = None,
) -> tuple[dict | None, str, str | None]:
    """Resolves user profile and checks tenant RBAC permission if requested.

    Returns `(user_record, authorized_tenant, error_message)`.
    """
    uid = user.get("uid")
    user_record = get_user_record(uid, user.get("email", ""), user.get("id_token"))
    if not user_record:
        return None, "", "Access Denied: Your User ID has not been registered in the database."

    authorized_tenant = user_record.get("tenant_id", "alpha")
    if required_permission:
        authorized, err_msg = verify_tenant_access(user_record, authorized_tenant, required_permission)
        if not authorized:
            return user_record, authorized_tenant, err_msg

    return user_record, authorized_tenant, None


def require_tenant_permission(permission: str):
    """Dependency factory enforcing authentication and strict single-tenant permission."""

    def _dependency(user: dict = Depends(get_current_user)) -> dict:
        user_record, authorized_tenant, err_msg = resolve_authorized_user(user, permission)
        if err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        return {
            "user": user,
            "user_record": user_record,
            "tenant_id": authorized_tenant,
        }

    return _dependency


def get_authorized_document(doc_id: str, authorized_tenant: str) -> dict:
    """Fetches a document record and enforces strict ownership by the authorized tenant."""
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")
    if doc.get("tenant_id") != authorized_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: Document '{doc_id}' belongs to another tenant.",
        )
    return doc
