import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import requests
from app.config import FIREBASE_PROJECT_ID, BASE_DIR

USERS_STORE_FILE = BASE_DIR / "app" / "data" / "firebase_users.json"
DOCUMENTS_STORE_FILE = BASE_DIR / "app" / "data" / "firebase_documents.json"

DEFAULT_PERMISSIONS = ["READ", "WRITE", "UPDATE", "DELETE", "INGEST"]


# ==========================================
# User Storage & Dynamic Firestore Sync
# ==========================================

def _load_local_users() -> dict:
    if USERS_STORE_FILE.exists():
        try:
            with open(USERS_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_local_users(data: dict):
    USERS_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_dynamic_tenant_id(email: str, uid: str) -> str:
    """Generates a clean dynamic tenant identifier from email or UID."""
    if email and "@" in email:
        username = email.split("@")[0]
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", username).strip("_").lower()
        if clean:
            return clean
    return f"tenant_{uid[:8].lower()}"


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user record from Firebase Firestore.
    If the user does not exist in Firestore yet, AUTOMATICALLY creates their record
    with their UID, email, and dynamic tenant_id without manual entry.
    """
    local_data = _load_local_users()
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Query Firestore REST API if id_token provided
    firestore_found = False
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}"}
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code == 200:
                fields = resp.json().get("fields", {})
                tenant = fields.get("tenant_id", {}).get("stringValue", "").strip()
                if not tenant:
                    tenant = _generate_dynamic_tenant_id(email, uid)
                is_active = fields.get("is_active", {}).get("booleanValue", True)
                perms_array = fields.get("permissions", {}).get("arrayValue", {}).get("values", [])
                perms = [p.get("stringValue").upper() for p in perms_array if p.get("stringValue")] or DEFAULT_PERMISSIONS
                user_email = fields.get("email", {}).get("stringValue", email)

                firestore_record = {
                    "uid": uid,
                    "user_uuid": uid,
                    "email": user_email,
                    "tenant_id": tenant,
                    "is_active": is_active,
                    "permissions": perms,
                    "last_login": now_iso,
                }
                local_data[uid] = firestore_record
                _save_local_users(local_data)
                return firestore_record

            elif resp.status_code == 404:
                # Document does not exist yet -> We will automatically create it below
                firestore_found = False

        except Exception as e:
            print(f"[FirebaseDB] Firestore lookup note: {e}")

    # 2. Check local store if user was already created
    if uid in local_data:
        record = local_data[uid]
        record["last_login"] = now_iso
        if email and not record.get("email"):
            record["email"] = email
        if not record.get("tenant_id"):
            record["tenant_id"] = _generate_dynamic_tenant_id(email, uid)
        _save_local_users(local_data)
        return record

    # 3. AUTOMATIC CREATION: User logs in for the first time
    dynamic_tenant = _generate_dynamic_tenant_id(email, uid)
    new_record = {
        "uid": uid,
        "user_uuid": uid,
        "email": email,
        "tenant_id": dynamic_tenant,
        "is_active": True,
        "permissions": DEFAULT_PERMISSIONS,
        "created_at": now_iso,
        "last_login": now_iso,
    }

    # Automatically persist to Firebase Firestore
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            payload = {
                "fields": {
                    "email": {"stringValue": email or ""},
                    "tenant_id": {"stringValue": dynamic_tenant},
                    "is_active": {"booleanValue": True},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in DEFAULT_PERMISSIONS]}},
                }
            }
            resp = requests.patch(
                url,
                headers={"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"},
                json=payload,
                timeout=5,
            )
            print(f"[FirebaseDB] Automatically created user '{uid}' in Firestore with tenant '{dynamic_tenant}' (status {resp.status_code})")
        except Exception as err:
            print(f"[FirebaseDB] Auto-create Firestore note: {err}")

    # Save to local cache
    local_data[uid] = new_record
    _save_local_users(local_data)
    return new_record


def verify_tenant_access(user_record: dict | None, target_tenant: str, required_permission: str) -> tuple[bool, str]:
    """Enforces:
    1. User record exists.
    2. User account is active.
    3. User belongs to requested tenant.
    4. User possesses the required permission.
    """
    if not user_record:
        return (False, "Access Denied: User profile could not be loaded.")

    if not user_record.get("is_active", True):
        return (False, "Account Suspended: Your user account is inactive. Please contact your administrator.")

    user_tenant = user_record.get("tenant_id")
    if not user_tenant:
        return (False, "Configuration Error: No tenant assigned to your account.")

    # Single Tenant Isolation Check
    if target_tenant != user_tenant:
        return (
            False,
            f"Tenant Authorisation Failed: You are only authorized to access tenant '{user_tenant}'.",
        )

    raw_perms = user_record.get("permissions", [])
    user_perms = [p.upper() for p in raw_perms]
    req_perm = required_permission.upper()

    if req_perm not in user_perms:
        return (
            False,
            f"Permission Denied: User lacks the '{req_perm}' permission on tenant '{target_tenant}'. Required: {req_perm}.",
        )

    return (True, "")


# ==========================================
# Document Metadata Storage (Firebase DB)
# ==========================================

def _load_local_documents() -> dict:
    if DOCUMENTS_STORE_FILE.exists():
        try:
            with open(DOCUMENTS_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_local_documents(data: dict):
    DOCUMENTS_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_tenant_documents(tenant_id: str) -> list[dict]:
    """Retrieves all documents strictly belonging to the given tenant."""
    docs = _load_local_documents()
    tenant_docs = [
        d for d in docs.values()
        if d.get("tenant_id") == tenant_id
    ]
    # Sort by upload timestamp descending (newest first)
    tenant_docs.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)
    return tenant_docs


def get_document_by_id(doc_id: str) -> dict | None:
    """Retrieves a document record by its UUID."""
    docs = _load_local_documents()
    return docs.get(doc_id)


def create_document_record(doc_data: dict, id_token: str | None = None) -> dict:
    """Creates a new document record in Firebase Database."""
    docs = _load_local_documents()
    doc_id = doc_data.get("doc_id") or str(uuid.uuid4())
    doc_data["doc_id"] = doc_id
    docs[doc_id] = doc_data
    _save_local_documents(docs)

    # Sync to Firestore if token available
    if FIREBASE_PROJECT_ID and id_token:
        try:
            tenant_id = doc_data.get("tenant_id", "default")
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
            fields = {
                "doc_id": {"stringValue": doc_id},
                "filename": {"stringValue": doc_data.get("filename", "")},
                "file_type": {"stringValue": doc_data.get("file_type", "")},
                "tenant_id": {"stringValue": tenant_id},
                "owner_uid": {"stringValue": doc_data.get("owner_uid", "")},
                "status": {"stringValue": doc_data.get("status", "uploaded")},
                "chunk_count": {"integerValue": str(doc_data.get("chunk_count", 0))},
                "upload_timestamp": {"stringValue": doc_data.get("upload_timestamp", "")},
            }
            requests.patch(url, headers={"Authorization": f"Bearer {id_token}"}, json={"fields": fields}, timeout=3)
        except Exception as e:
            print(f"[FirebaseDB] Firestore doc sync note: {e}")

    return doc_data


def update_document_record(doc_id: str, updates: dict, id_token: str | None = None) -> dict | None:
    """Updates an existing document record."""
    docs = _load_local_documents()
    if doc_id not in docs:
        return None

    docs[doc_id].update(updates)
    _save_local_documents(docs)

    # Sync to Firestore if token available
    if FIREBASE_PROJECT_ID and id_token:
        try:
            tenant_id = docs[doc_id].get("tenant_id", "default")
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
            fields = {}
            for k, v in updates.items():
                if isinstance(v, str):
                    fields[k] = {"stringValue": v}
                elif isinstance(v, int):
                    fields[k] = {"integerValue": str(v)}
                elif isinstance(v, bool):
                    fields[k] = {"booleanValue": v}
            if fields:
                requests.patch(url, headers={"Authorization": f"Bearer {id_token}"}, json={"fields": fields}, timeout=3)
        except Exception:
            pass

    return docs[doc_id]


def delete_document_record(doc_id: str, id_token: str | None = None) -> bool:
    """Permanently deletes a document record from Firebase Database."""
    docs = _load_local_documents()
    if doc_id in docs:
        tenant_id = docs[doc_id].get("tenant_id", "default")
        del docs[doc_id]
        _save_local_documents(docs)

        # Sync deletion to Firestore
        if FIREBASE_PROJECT_ID and id_token:
            try:
                url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
                requests.delete(url, headers={"Authorization": f"Bearer {id_token}"}, timeout=3)
            except Exception:
                pass
        return True
    return False
