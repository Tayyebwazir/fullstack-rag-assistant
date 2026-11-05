import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from app.config import FIREBASE_PROJECT_ID, BASE_DIR

USERS_STORE_FILE = BASE_DIR / "app" / "data" / "firebase_users.json"
DOCUMENTS_STORE_FILE = BASE_DIR / "app" / "data" / "firebase_documents.json"
CREDENTIALS_FILE = BASE_DIR / "app" / "data" / "firebase_credentials.json"
FALLBACK_CREDENTIALS_FILE = Path(r"D:\DB_config\authen-81a29-firebase-adminsdk-fbsvc-457a3fc98f.json")

DEFAULT_PERMISSIONS = ["READ", "WRITE", "UPDATE", "DELETE", "INGEST"]

_firestore_client = None

def _get_firestore_client():
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    try:
        if not firebase_admin._apps:
            cred_file = None
            if CREDENTIALS_FILE.exists():
                cred_file = CREDENTIALS_FILE
            elif FALLBACK_CREDENTIALS_FILE.exists():
                cred_file = FALLBACK_CREDENTIALS_FILE

            if cred_file:
                cred = credentials.Certificate(str(cred_file))
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()
        _firestore_client = firestore.client()
        return _firestore_client
    except Exception as e:
        print(f"[FirebaseAdmin] Firestore client init note: {e}")
        return None


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


def _is_valid_weaviate_tenant(tenant: str | None, active_tenants: list[str]) -> bool:
    if not tenant or not isinstance(tenant, str):
        return False
    t = tenant.strip().lower()
    # Reject legacy strings and prefix templates
    if t in ("alpha", "beta", "tri", "default") or t.startswith("tenant_"):
        return False
    return tenant.strip() in active_tenants


def _assign_unique_tenant_for_user(uid: str, local_data: dict, current_fs_tenant: str = "") -> str:
    """Assigns a strictly UNIQUE tenant shard for each user directly from Weaviate DB.
    Guarantees no two users share the same tenant.
    Always uses real Weaviate UUID tenant IDs.
    """
    try:
        from app.services.vector_store import get_weaviate_tenants, create_weaviate_tenant
        weaviate_tenants = get_weaviate_tenants()
    except Exception as e:
        print(f"[Weaviate] Error getting tenants: {e}")
        weaviate_tenants = []

    # Collect all tenants currently claimed by OTHER users in local store
    other_assigned_tenants = {
        user_info.get("tenant_id") for other_uid, user_info in local_data.items()
        if other_uid != uid and _is_valid_weaviate_tenant(user_info.get("tenant_id"), weaviate_tenants)
    }

    # Also collect tenants currently claimed by other users in Firestore
    try:
        db = _get_firestore_client()
        if db:
            for doc in db.collection("users").stream():
                if doc.id != uid:
                    t = (doc.to_dict() or {}).get("tenant_id")
                    if _is_valid_weaviate_tenant(t, weaviate_tenants):
                        other_assigned_tenants.add(t)
    except Exception:
        pass

    # If this user already has a valid tenant from Weaviate that is not claimed by anyone else, keep it
    candidate = current_fs_tenant or local_data.get(uid, {}).get("tenant_id")
    if _is_valid_weaviate_tenant(candidate, weaviate_tenants) and candidate not in other_assigned_tenants:
        return candidate.strip()

    # Look for an available, unassigned tenant in Weaviate (e.g., db261c85-2d6f-4a93-8aeb-1f725a5e30cc)
    available = [
        t for t in weaviate_tenants
        if t not in other_assigned_tenants and _is_valid_weaviate_tenant(t, weaviate_tenants)
    ]

    if available:
        available.sort()
        return available[0]

    # If all existing Weaviate tenants are taken (or no tenant exists), dynamically create a new UUID tenant in Weaviate!
    new_tenant = str(uuid.uuid4())
    try:
        from app.services.vector_store import create_weaviate_tenant
        create_weaviate_tenant(new_tenant)
        print(f"[Weaviate] Dynamically provisioned new UUID tenant shard: '{new_tenant}'")
    except Exception as e:
        print(f"[Weaviate] Note provisioning shard {new_tenant}: {e}")

    return new_tenant


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user record and guarantees a strictly UNIQUE Weaviate tenant shard per user.
    Always syncs the unique tenant assignment directly into Firestore under users/{uid}.
    """
    local_data = _load_local_users()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Check existing Firestore document first if available
    fs_tenant = ""
    fs_email = ""
    fs_record = {}
    try:
        db = _get_firestore_client()
        if db:
            doc_snap = db.collection("users").document(uid).get()
            if doc_snap.exists:
                fs_record = doc_snap.to_dict() or {}
                fs_tenant = fs_record.get("tenant_id", "")
                fs_email = fs_record.get("email", "")
    except Exception as e:
        print(f"[FirebaseDB] Firestore read note: {e}")

    # Assign a unique, isolated tenant for this user directly from Weaviate
    user_tenant = _assign_unique_tenant_for_user(uid, local_data, fs_tenant)

    effective_email = email or fs_email or local_data.get(uid, {}).get("email", "")

    record = {
        "uid": uid,
        "user_uuid": uid,
        "email": effective_email,
        "tenant_id": user_tenant,
        "is_active": fs_record.get("is_active", local_data.get(uid, {}).get("is_active", True)),
        "permissions": fs_record.get("permissions", local_data.get(uid, {}).get("permissions", DEFAULT_PERMISSIONS)),
        "created_at": fs_record.get("created_at", local_data.get(uid, {}).get("created_at", now_iso)),
        "last_login": now_iso,
    }

    # 1. Direct sync into Firestore via Admin SDK
    try:
        db = _get_firestore_client()
        if db:
            db.collection("users").document(uid).set({
                "email": record["email"],
                "tenant_id": user_tenant,
                "is_active": record["is_active"],
                "permissions": record["permissions"],
                "created_at": record["created_at"],
                "last_login": record["last_login"],
            }, merge=True)
            print(f"[FirebaseDB Admin] Synced Firestore users/{uid} with tenant_id '{user_tenant}'")
    except Exception as e:
        print(f"[FirebaseDB Admin] Sync error: {e}")

    # 2. Sync via REST API as extra guarantee if id_token provided
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
            payload = {
                "fields": {
                    "email": {"stringValue": record["email"]},
                    "tenant_id": {"stringValue": user_tenant},
                    "is_active": {"booleanValue": record["is_active"]},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in record["permissions"]]}},
                }
            }
            resp = requests.patch(url, headers=headers, json=payload, timeout=5)
            print(f"[FirebaseDB REST] Synced Firestore users/{uid} with unique tenant '{user_tenant}' (status {resp.status_code})")
        except Exception as e:
            print(f"[FirebaseDB REST] Sync note: {e}")

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
    tenant_id = doc_data.get("tenant_id", "default")
    docs[doc_id] = doc_data
    _save_local_documents(docs)

    # 1. Direct Firestore Admin SDK sync
    try:
        db = _get_firestore_client()
        if db:
            db.collection(f"tenants_{tenant_id}_docs").document(doc_id).set(doc_data)
    except Exception as e:
        print(f"[FirebaseDB Admin] Doc create sync note: {e}")

    # 2. REST API fallback
    if FIREBASE_PROJECT_ID and id_token:
        try:
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
    tenant_id = docs[doc_id].get("tenant_id", "default")
    _save_local_documents(docs)

    # 1. Direct Firestore Admin SDK sync
    try:
        db = _get_firestore_client()
        if db:
            db.collection(f"tenants_{tenant_id}_docs").document(doc_id).set(updates, merge=True)
    except Exception as e:
        print(f"[FirebaseDB Admin] Doc update sync note: {e}")

    # 2. REST API fallback
    if FIREBASE_PROJECT_ID and id_token:
        try:
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

        # 1. Direct Firestore Admin SDK sync
        try:
            db = _get_firestore_client()
            if db:
                db.collection(f"tenants_{tenant_id}_docs").document(doc_id).delete()
        except Exception as e:
            print(f"[FirebaseDB Admin] Doc delete sync note: {e}")

        # 2. REST API fallback
        if FIREBASE_PROJECT_ID and id_token:
            try:
                url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
                requests.delete(url, headers={"Authorization": f"Bearer {id_token}"}, timeout=3)
            except Exception:
                pass
        return True
    return False
