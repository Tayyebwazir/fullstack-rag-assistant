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

    weaviate_tenants.sort()

    # Distribute users evenly across authentic Weaviate tenants
    counts = {t: 0 for t in weaviate_tenants}
    for u in local_data.values():
        t = u.get("tenant_id")
        if t in counts:
            counts[t] += 1

    assigned_tenant = min(counts, key=counts.get)
    return assigned_tenant


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user record from Firebase Firestore.
    Assigns tenant_id directly from Weaviate DB (not email) and stores it in Firestore.
    """
    local_data = _load_local_users()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        from app.services.vector_store import get_weaviate_tenants
        active_weaviate_tenants = get_weaviate_tenants()
    except Exception:
        active_weaviate_tenants = ["alpha", "beta", "tri"]

    # 1. Query Firestore REST API if id_token provided
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}"}
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code == 200:
                fields = resp.json().get("fields", {})
                tenant = fields.get("tenant_id", {}).get("stringValue", "").strip()

                # If tenant in Firestore is missing or was mistakenly created from email, fix it using Weaviate DB
                if not tenant or tenant not in active_weaviate_tenants:
                    tenant = _get_tenant_from_weaviate(local_data)
                    # Update Firestore with the real Weaviate tenant
                    try:
                        patch_payload = {
                            "fields": {
                                "email": {"stringValue": fields.get("email", {}).get("stringValue", email)},
                                "tenant_id": {"stringValue": tenant},
                                "is_active": fields.get("is_active", {"booleanValue": True}),
                                "permissions": fields.get("permissions", {"arrayValue": {"values": [{"stringValue": p} for p in DEFAULT_PERMISSIONS]}}),
                            }
                        }
                        requests.patch(url, headers=headers, json=patch_payload, timeout=3)
                        print(f"[FirebaseDB] Updated Firestore user '{uid}' with Weaviate tenant: '{tenant}'")
                    except Exception as pe:
                        print(f"[FirebaseDB] Patch note: {pe}")

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

        except Exception as e:
            print(f"[FirebaseDB] Firestore lookup note: {e}")

    # 2. Check local store if user was already created
    if uid in local_data:
        record = local_data[uid]
        record["last_login"] = now_iso
        if email and not record.get("email"):
            record["email"] = email
        if not record.get("tenant_id") or record.get("tenant_id") not in active_weaviate_tenants:
            record["tenant_id"] = _get_tenant_from_weaviate(local_data)
        _save_local_users(local_data)
        return record

    # 3. AUTOMATIC CREATION: User logs in for the first time
    # Tenant is fetched directly from Weaviate DB
    weaviate_tenant = _get_tenant_from_weaviate(local_data)
    new_record = {
        "uid": uid,
        "user_uuid": uid,
        "email": email,
        "tenant_id": weaviate_tenant,
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
                    "tenant_id": {"stringValue": weaviate_tenant},
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
            print(f"[FirebaseDB] Created user '{uid}' in Firestore with Weaviate tenant '{weaviate_tenant}' (status {resp.status_code})")
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
