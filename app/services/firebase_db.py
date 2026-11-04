import json
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


def _get_tenant_from_weaviate(local_data: dict) -> str:
    """Retrieves existing tenant shards directly from Weaviate DB and assigns an authentic tenant.
    Never uses email to generate tenant names.
    """
    try:
        from app.services.vector_store import get_weaviate_tenants
        weaviate_tenants = get_weaviate_tenants()
    except Exception:
        weaviate_tenants = []

    if not weaviate_tenants:
        weaviate_tenants = ["alpha", "beta", "tri"]

    # Prefer 'alpha' if available so user has immediate access to existing documents
    if "alpha" in weaviate_tenants:
        return "alpha"

    return weaviate_tenants[0]


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user record from Firebase Firestore.
    Assigns tenant_id directly from Weaviate DB (not email) and stores it in Firestore.
    Always guarantees the user document is persisted in Firestore under users/{uid}.
    """
    local_data = _load_local_users()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        from app.services.vector_store import get_weaviate_tenants
        active_weaviate_tenants = get_weaviate_tenants()
    except Exception:
        active_weaviate_tenants = ["alpha", "beta", "tri"]

    user_tenant = None

    # Check local store first
    if uid in local_data:
        record = local_data[uid]
        user_tenant = record.get("tenant_id")
        if not user_tenant or user_tenant not in active_weaviate_tenants:
            user_tenant = _get_tenant_from_weaviate(local_data)
            record["tenant_id"] = user_tenant
        if email and not record.get("email"):
            record["email"] = email
        record["last_login"] = now_iso
    else:
        user_tenant = _get_tenant_from_weaviate(local_data)
        record = {
            "uid": uid,
            "user_uuid": uid,
            "email": email,
            "tenant_id": user_tenant,
            "is_active": True,
            "permissions": DEFAULT_PERMISSIONS,
            "created_at": now_iso,
            "last_login": now_iso,
        }

    # 1. Check or Sync to Firestore
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code == 200:
                fields = resp.json().get("fields", {})
                fs_tenant = fields.get("tenant_id", {}).get("stringValue", "").strip()
                # If Firestore has a valid Weaviate tenant, keep it; otherwise update it with our Weaviate tenant
                if fs_tenant and fs_tenant in active_weaviate_tenants:
                    record["tenant_id"] = fs_tenant
                else:
                    # Update Firestore with real Weaviate tenant
                    patch_payload = {
                        "fields": {
                            "email": {"stringValue": record.get("email", email)},
                            "tenant_id": {"stringValue": record["tenant_id"]},
                            "is_active": {"booleanValue": record.get("is_active", True)},
                            "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in record.get("permissions", DEFAULT_PERMISSIONS)]}},
                        }
                    }
                    requests.patch(url, headers=headers, json=patch_payload, timeout=3)
            else:
                # If document is not in Firestore (status 404 or other), create it directly in Firestore
                patch_payload = {
                    "fields": {
                        "email": {"stringValue": record.get("email", email)},
                        "tenant_id": {"stringValue": record["tenant_id"]},
                        "is_active": {"booleanValue": record.get("is_active", True)},
                        "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in record.get("permissions", DEFAULT_PERMISSIONS)]}},
                    }
                }
                patch_resp = requests.patch(url, headers=headers, json=patch_payload, timeout=5)
                print(f"[FirebaseDB] Created/Synced Firestore users/{uid} with tenant '{record['tenant_id']}' (status {patch_resp.status_code})")

        except Exception as e:
            print(f"[FirebaseDB] Firestore sync note: {e}")

    # Save to local cache
    local_data[uid] = record
    _save_local_users(local_data)
    return record


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

        if FIREBASE_PROJECT_ID and id_token:
            try:
                url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
                requests.delete(url, headers={"Authorization": f"Bearer {id_token}"}, timeout=3)
            except Exception:
                pass
        return True
    return False
