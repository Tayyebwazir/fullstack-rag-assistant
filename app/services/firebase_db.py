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


def _assign_unique_tenant_for_user(uid: str, local_data: dict) -> str:
    """Assigns a strictly UNIQUE tenant shard for each user directly from Weaviate DB.
    Guarantees no two users share the same tenant.
    """
    try:
        from app.services.vector_store import get_weaviate_tenants, connect_weaviate, COLLECTION_NAME
        from weaviate.classes.tenants import Tenant
        weaviate_tenants = get_weaviate_tenants()
    except Exception:
        weaviate_tenants = []

    if not weaviate_tenants:
        weaviate_tenants = ["alpha", "beta", "tri"]

    # Collect all tenants currently assigned to OTHER users
    other_assigned_tenants = {
        user_info.get("tenant_id") for other_uid, user_info in local_data.items()
        if other_uid != uid and user_info.get("tenant_id")
    }

    # If this user already has an assigned tenant that no one else has, keep it
    current_user_tenant = local_data.get(uid, {}).get("tenant_id")
    if current_user_tenant and current_user_tenant not in other_assigned_tenants:
        return current_user_tenant

    # Find a Weaviate tenant shard not taken by any other user
    available = [t for t in weaviate_tenants if t not in other_assigned_tenants]

    if available:
        available.sort()
        return available[0]

    # If all existing Weaviate shards are assigned, dynamically create a new tenant shard in Weaviate!
    new_tenant = f"tenant_{uid[:8].lower()}"
    try:
        with connect_weaviate() as client:
            if client.collections.exists(COLLECTION_NAME):
                col = client.collections.get(COLLECTION_NAME)
                col.tenants.create([Tenant(name=new_tenant)])
                print(f"[Weaviate] Dynamically provisioned dedicated tenant shard: '{new_tenant}'")
    except Exception as e:
        print(f"[Weaviate] Note provisioning shard {new_tenant}: {e}")

    return new_tenant


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user record and guarantees a strictly UNIQUE Weaviate tenant shard per user.
    Always syncs the unique tenant assignment directly into Firestore under users/{uid}.
    """
    local_data = _load_local_users()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Assign a unique, isolated tenant for this user
    user_tenant = _assign_unique_tenant_for_user(uid, local_data)

    if uid in local_data:
        record = local_data[uid]
        record["tenant_id"] = user_tenant
        if email and not record.get("email"):
            record["email"] = email
        record["last_login"] = now_iso
    else:
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

    # Always sync to Firestore so the user document is up to date and visible in console
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
            payload = {
                "fields": {
                    "email": {"stringValue": record.get("email", email)},
                    "tenant_id": {"stringValue": user_tenant},
                    "is_active": {"booleanValue": record.get("is_active", True)},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in record.get("permissions", DEFAULT_PERMISSIONS)]}},
                }
            }
            resp = requests.patch(url, headers=headers, json=payload, timeout=5)
            print(f"[FirebaseDB] Synced Firestore users/{uid} with unique tenant '{user_tenant}' (status {resp.status_code})")
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

    # Strict 1-to-1 Tenant Isolation Check
    if target_tenant != user_tenant:
        return (
            False,
            f"Tenant Authorisation Failed: You are only authorized to access tenant '{user_tenant}'. Cross-tenant access is strictly forbidden.",
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
