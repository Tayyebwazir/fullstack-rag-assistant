import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

security = HTTPBearer(auto_error=False)

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "authen-81a29")
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
