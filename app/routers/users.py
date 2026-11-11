"""User profile and Firebase tenant authorization routes."""

from fastapi import APIRouter, Depends, HTTPException

from app.services.auth import get_current_user
from app.services.firebase_db import get_user_record

router = APIRouter(prefix="/user", tags=["Users"])


@router.get("/profile")
def get_user_profile(user: dict = Depends(get_current_user)):
    """Fetches user UUID, assigned tenant ID, active status, and permissions from Firebase Database."""
    uid = user.get("uid")
    email = user.get("email", "")
    id_token = user.get("id_token")
    profile = get_user_record(uid, email, id_token)
    if not profile:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Your User ID has not been registered in the database. Please contact an administrator to assign you a tenant.",
        )
    if not profile.get("is_active", True):
        raise HTTPException(
            status_code=403,
            detail="Account Inactive: Your account has been suspended.",
        )
    return {
        "status": "success",
        "profile": profile,
        "assigned_tenant": profile.get("tenant_id", "alpha"),
    }
