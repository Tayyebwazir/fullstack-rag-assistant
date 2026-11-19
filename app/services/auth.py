import logging
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

from app.config import FIREBASE_PROJECT_ID
google_request_adapter = google_requests.Request()


def get_current_user(
    auth_header: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """
    FastAPI dependency that cryptographically verifies the Firebase JWT ID token
    using Google's official public certificate endpoints.
    Extracts the user's UID and provides authenticated context.
    """
    if not auth_header or not auth_header.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in via Firebase to access this resource.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.credentials

    if not FIREBASE_PROJECT_ID:
        logger.error("FIREBASE_PROJECT_ID is not configured in environment variables.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service configuration error.",
        )

    try:
        # Cryptographically verifies token signature, expiration, issuer and audience
        decoded_token = id_token.verify_firebase_token(
            id_token=token,
            request=google_request_adapter,
            audience=FIREBASE_PROJECT_ID,
        )

        uid = decoded_token.get("user_id") or decoded_token.get("sub")
        if not uid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user UID.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "uid": uid,
            "email": decoded_token.get("email", ""),
            "name": decoded_token.get("name", ""),
            "picture": decoded_token.get("picture", ""),
            "id_token": token,
        }
    except ValueError as exc:
        logger.warning("Token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected authentication error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Unable to verify credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )