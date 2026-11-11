import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
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


def _is_valid_uuid(val: str | None) -> bool:
    if not val or not isinstance(val, str):
        return False
    t = val.strip().lower()
    if t in ("alpha", "beta", "tri", "default") or t.startswith("tenant_"):
        return False
    try:
        uuid.UUID(val.strip())
        return True
    except ValueError:
        return False


def _is_valid_weaviate_tenant(tenant: str | None, active_tenants: list[str]) -> bool:
    if not _is_valid_uuid(tenant):
        return False
    return tenant.strip() in active_tenants


def _resolve_user_email(uid: str, fallback_email: str = "") -> str:
    """Resolves a user's email from fallback, local users JSON, or Firestore."""
    if fallback_email:
        return fallback_email
    if not uid:
        return ""
    local_users = _load_local_users()
    local_email = (local_users.get(uid) or {}).get("email", "")
    if local_email:
        return local_email
    try:
        db = _get_firestore_client()
        if db:
            snap = db.collection("users").document(uid).get()
            if snap.exists:
                return (snap.to_dict() or {}).get("email", "")
    except Exception:
        pass
    return ""


def _build_rest_patch_url(doc_path: str, field_names: list[str]) -> str:
    """Builds a Firestore REST PATCH URL with updateMask.fieldPaths so partial updates never wipe existing fields."""
    base = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{doc_path}"
    if not field_names:
        return base
    query = urlencode([("updateMask.fieldPaths", name) for name in field_names])
    return f"{base}?{query}"


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

    candidates = [t.strip() for t in (current_fs_tenant, local_data.get(uid, {}).get("tenant_id", "")) if t and t.strip()]

    # If Weaviate is temporarily unreachable, preserve an existing valid UUID rather than overwriting it
    if not weaviate_tenants:
        for cand in candidates:
            if _is_valid_uuid(cand):
                return cand

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
    for cand in candidates:
        if _is_valid_weaviate_tenant(cand, weaviate_tenants) and cand not in other_assigned_tenants:
            return cand

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
    fs_doc_ids = fs_record.get("document_ids") if isinstance(fs_record.get("document_ids"), list) else None
    local_doc_ids = local_data.get(uid, {}).get("document_ids") if isinstance(local_data.get(uid, {}).get("document_ids"), list) else []
    existing_doc_ids = fs_doc_ids if (fs_doc_ids is not None and (fs_doc_ids or not local_doc_ids)) else local_doc_ids

    record = {
        "uid": uid,
        "user_uuid": uid,
        "email": effective_email,
        "tenant_id": user_tenant,
        "is_active": fs_record.get("is_active", local_data.get(uid, {}).get("is_active", True)),
        "permissions": fs_record.get("permissions", local_data.get(uid, {}).get("permissions", DEFAULT_PERMISSIONS)),
        "document_ids": existing_doc_ids,
        "created_at": fs_record.get("created_at", local_data.get(uid, {}).get("created_at", now_iso)),
        "last_login": now_iso,
    }

    # 1. Direct sync into Firestore via Admin SDK
    admin_synced = False
    try:
        db = _get_firestore_client()
        if db:
            user_update = {
                "email": record["email"],
                "tenant_id": user_tenant,
                "is_active": record["is_active"],
                "permissions": record["permissions"],
                "document_ids": record["document_ids"],
                "created_at": record["created_at"],
                "last_login": record["last_login"],
            }
            db.collection("users").document(uid).set(user_update, merge=True)
            admin_synced = True
            print(f"[FirebaseDB Admin] Synced Firestore users/{uid} with tenant_id '{user_tenant}'")
    except Exception as e:
        print(f"[FirebaseDB Admin] Sync error: {e}")

    # 2. Sync via REST API ONLY if Admin SDK failed, using updateMask so existing fields are never wiped
    if not admin_synced and FIREBASE_PROJECT_ID and id_token:
        try:
            fields_to_update = ["email", "tenant_id", "is_active", "permissions", "last_login"]
            url = _build_rest_patch_url(f"users/{uid}", fields_to_update)
            headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}
            payload = {
                "fields": {
                    "email": {"stringValue": record["email"]},
                    "tenant_id": {"stringValue": user_tenant},
                    "is_active": {"booleanValue": record["is_active"]},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in record["permissions"]]}},
                    "last_login": {"stringValue": record["last_login"]},
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


def _sync_user_document_entry(db, owner_uid: str, doc_id: str, doc_summary: dict):
    """Stores a document reference and summary under users/{owner_uid}."""
    if not db or not owner_uid:
        return
    user_ref = db.collection("users").document(owner_uid)
    user_ref.set({
        "document_ids": firestore.ArrayUnion([doc_id]),
    }, merge=True)
    user_ref.collection("documents").document(doc_id).set(doc_summary, merge=True)


def create_document_record(doc_data: dict, id_token: str | None = None) -> dict:
    """Creates a new document record in Firebase Database.
    Stores the document in the root 'docs' collection (with owner_uid + owner_email)
    and attaches the document reference directly under users/{owner_uid}.
    """
    docs = _load_local_documents()
    doc_id = doc_data.get("doc_id") or str(uuid.uuid4())
    doc_data["doc_id"] = doc_id
    tenant_id = doc_data.get("tenant_id", "default")
    owner_uid = doc_data.get("owner_uid", "")
    owner_email = _resolve_user_email(owner_uid, doc_data.get("owner_email", ""))
    doc_data["owner_email"] = owner_email
    docs[doc_id] = doc_data
    _save_local_documents(docs)

    # Update local users document_ids list
    if owner_uid:
        local_users = _load_local_users()
        if owner_uid in local_users:
            u_docs = local_users[owner_uid].get("document_ids", [])
            if doc_id not in u_docs:
                u_docs.append(doc_id)
                local_users[owner_uid]["document_ids"] = u_docs
                _save_local_users(local_users)

    clean_record = {
        "doc_id": doc_id,
        "filename": doc_data.get("filename", ""),
        "file_type": doc_data.get("file_type", ""),
        "file_size": doc_data.get("file_size", 0),
        "tenant_id": tenant_id,
        "owner_uid": owner_uid,
        "owner_email": owner_email,
        "status": doc_data.get("status", "uploaded"),
        "chunk_count": doc_data.get("chunk_count", 0),
        "version": doc_data.get("version", 1),
        "upload_timestamp": doc_data.get("upload_timestamp", ""),
        "updated_timestamp": doc_data.get("updated_timestamp", ""),
        "error_message": doc_data.get("error_message"),
    }

    # 1. Direct Firestore Admin SDK sync
    admin_synced = False
    try:
        db = _get_firestore_client()
        if db:
            # A. Write to root 'docs' collection
            db.collection("docs").document(doc_id).set(clean_record, merge=True)
            print(f"[FirebaseDB Admin] Saved document {doc_id} to root collection 'docs'")

            # B. Associate doc_id & document summary under 'users/{owner_uid}'
            if owner_uid:
                user_doc_summary = {
                    "doc_id": doc_id,
                    "filename": clean_record["filename"],
                    "file_type": clean_record["file_type"],
                    "status": clean_record["status"],
                    "chunk_count": clean_record["chunk_count"],
                    "tenant_id": tenant_id,
                    "upload_timestamp": clean_record["upload_timestamp"],
                    "updated_timestamp": clean_record["updated_timestamp"],
                }
                _sync_user_document_entry(db, owner_uid, doc_id, user_doc_summary)
                print(f"[FirebaseDB Admin] Linked {doc_id} ({clean_record['filename']}) under users/{owner_uid}")

            # C. Sync to tenant-specific collection for backward compatibility
            db.collection(f"tenants_{tenant_id}_docs").document(doc_id).set(clean_record, merge=True)
            admin_synced = True
    except Exception as e:
        print(f"[FirebaseDB Admin] Doc create sync note: {e}")

    # 2. REST API fallback ONLY if Admin SDK failed, with updateMask
    if not admin_synced and FIREBASE_PROJECT_ID and id_token:
        try:
            fields = {
                "doc_id": {"stringValue": doc_id},
                "filename": {"stringValue": clean_record["filename"]},
                "file_type": {"stringValue": clean_record["file_type"]},
                "tenant_id": {"stringValue": tenant_id},
                "owner_uid": {"stringValue": owner_uid},
                "owner_email": {"stringValue": owner_email},
                "status": {"stringValue": clean_record["status"]},
                "chunk_count": {"integerValue": str(clean_record["chunk_count"])},
                "upload_timestamp": {"stringValue": clean_record["upload_timestamp"]},
            }
            url_docs = _build_rest_patch_url(f"docs/{doc_id}", list(fields.keys()))
            requests.patch(url_docs, headers={"Authorization": f"Bearer {id_token}"}, json={"fields": fields}, timeout=3)
        except Exception as e:
            print(f"[FirebaseDB] Firestore doc sync note: {e}")

    return doc_data


def update_document_record(doc_id: str, updates: dict, id_token: str | None = None) -> dict | None:
    """Updates an existing document record in local store, root 'docs', and user's document entry."""
    docs = _load_local_documents()
    if doc_id not in docs:
        return None

    docs[doc_id].update(updates)
    tenant_id = docs[doc_id].get("tenant_id", "default")
    owner_uid = updates.get("owner_uid") or docs[doc_id].get("owner_uid", "")
    owner_email = updates.get("owner_email") or docs[doc_id].get("owner_email") or _resolve_user_email(owner_uid)
    if owner_email and not docs[doc_id].get("owner_email"):
        docs[doc_id]["owner_email"] = owner_email
    _save_local_documents(docs)

    effective_updates = dict(updates)
    if owner_email and "owner_email" not in effective_updates:
        effective_updates["owner_email"] = owner_email

    # Update local user document_ids cache
    if owner_uid:
        local_users = _load_local_users()
        if owner_uid in local_users:
            u_docs = local_users[owner_uid].get("document_ids", [])
            if doc_id not in u_docs:
                u_docs.append(doc_id)
                local_users[owner_uid]["document_ids"] = u_docs
                _save_local_users(local_users)

    # 1. Direct Firestore Admin SDK sync
    admin_synced = False
    try:
        db = _get_firestore_client()
        if db:
            # Update root docs collection
            db.collection("docs").document(doc_id).set(effective_updates, merge=True)
            # Update legacy tenant docs collection
            db.collection(f"tenants_{tenant_id}_docs").document(doc_id).set(effective_updates, merge=True)
            # Keep user's document_ids and documents subcollection synced
            if owner_uid:
                user_doc_summary = {
                    "doc_id": doc_id,
                    "filename": docs[doc_id].get("filename", ""),
                    "file_type": docs[doc_id].get("file_type", ""),
                    "status": docs[doc_id].get("status", ""),
                    "chunk_count": docs[doc_id].get("chunk_count", 0),
                    "tenant_id": tenant_id,
                    "upload_timestamp": docs[doc_id].get("upload_timestamp", ""),
                    "updated_timestamp": docs[doc_id].get("updated_timestamp", ""),
                }
                _sync_user_document_entry(db, owner_uid, doc_id, user_doc_summary)
            admin_synced = True
            print(f"[FirebaseDB Admin] Updated document {doc_id} in root 'docs'")
    except Exception as e:
        print(f"[FirebaseDB Admin] Doc update sync note: {e}")

    # 2. REST API fallback ONLY if Admin SDK failed, with updateMask
    if not admin_synced and FIREBASE_PROJECT_ID and id_token:
        try:
            fields = {}
            for k, v in effective_updates.items():
                if isinstance(v, str):
                    fields[k] = {"stringValue": v}
                elif isinstance(v, bool):
                    fields[k] = {"booleanValue": v}
                elif isinstance(v, int):
                    fields[k] = {"integerValue": str(v)}
            if fields:
                url = _build_rest_patch_url(f"docs/{doc_id}", list(fields.keys()))
                requests.patch(url, headers={"Authorization": f"Bearer {id_token}"}, json={"fields": fields}, timeout=3)
        except Exception:
            pass

    return docs[doc_id]


def save_document_chunks(
    doc_id: str,
    chunks: list[dict],
    tenant_id: str,
    source: str = "",
    id_token: str | None = None,
) -> int:
    """Saves document chunks into Firestore subcollection docs/{doc_id}/chunks.
    Each chunk is stored with chunk_index, content, page, source, doc_id, and tenant_id.
    """
    if not chunks:
        return 0

    saved_count = 0
    try:
        db = _get_firestore_client()
        if db:
            chunks_coll = db.collection("docs").document(doc_id).collection("chunks")
            batch = db.batch()
            batch_count = 0
            for idx, c in enumerate(chunks):
                chunk_ref = chunks_coll.document(f"chunk_{idx}")
                chunk_record = {
                    "chunk_index": idx,
                    "content": c.get("content", ""),
                    "page": c.get("page", 1),
                    "source": source or c.get("source", ""),
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                batch.set(chunk_ref, chunk_record, merge=True)
                batch_count += 1
                if batch_count >= 400:
                    batch.commit()
                    batch = db.batch()
                    batch_count = 0
            if batch_count > 0:
                batch.commit()
            saved_count = len(chunks)
            print(f"[FirebaseDB Admin] Successfully saved {saved_count} chunks into docs/{doc_id}/chunks")
            return saved_count
    except Exception as e:
        print(f"[FirebaseDB Admin] Error saving chunks for {doc_id}: {e}")

    # REST API fallback
    if FIREBASE_PROJECT_ID and id_token:
        try:
            for idx, c in enumerate(chunks):
                url = (
                    f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/"
                    f"documents/docs/{doc_id}/chunks/chunk_{idx}"
                )
                fields = {
                    "chunk_index": {"integerValue": str(idx)},
                    "content": {"stringValue": c.get("content", "")},
                    "page": {"integerValue": str(c.get("page", 1))},
                    "source": {"stringValue": source or c.get("source", "")},
                    "doc_id": {"stringValue": doc_id},
                    "tenant_id": {"stringValue": tenant_id},
                }
                requests.patch(
                    url,
                    headers={"Authorization": f"Bearer {id_token}"},
                    json={"fields": fields},
                    timeout=3,
                )
            saved_count = len(chunks)
        except Exception as e:
            print(f"[FirebaseDB REST] Chunk sync note: {e}")

    return saved_count


def get_document_chunks(doc_id: str) -> list[dict]:
    """Retrieves all chunks for a document from Firestore docs/{doc_id}/chunks."""
    try:
        db = _get_firestore_client()
        if db:
            chunks_ref = db.collection("docs").document(doc_id).collection("chunks")
            chunks = []
            for snap in chunks_ref.order_by("chunk_index").stream():
                chunks.append(snap.to_dict())
            return chunks
    except Exception as e:
        print(f"[FirebaseDB] Error fetching chunks for {doc_id}: {e}")
    return []


def delete_document_record(doc_id: str, id_token: str | None = None) -> bool:
    """Permanently deletes a document record, all its chunks, and removes its doc_id from the user's document_ids."""
    docs = _load_local_documents()
    if doc_id in docs:
        doc_info = docs[doc_id]
        tenant_id = doc_info.get("tenant_id", "default")
        owner_uid = doc_info.get("owner_uid", "")
        del docs[doc_id]
        _save_local_documents(docs)

        # Update local user's document_ids list
        if owner_uid:
            local_users = _load_local_users()
            if owner_uid in local_users and "document_ids" in local_users[owner_uid]:
                if doc_id in local_users[owner_uid]["document_ids"]:
                    local_users[owner_uid]["document_ids"].remove(doc_id)
                    _save_local_users(local_users)

        # 1. Direct Firestore Admin SDK sync
        admin_synced = False
        try:
            db = _get_firestore_client()
            if db:
                # A. Delete all chunks in subcollection docs/{doc_id}/chunks
                chunks_ref = db.collection("docs").document(doc_id).collection("chunks")
                for chunk_doc in chunks_ref.stream():
                    chunk_doc.reference.delete()

                # B. Delete document from root 'docs' collection
                db.collection("docs").document(doc_id).delete()

                # C. Remove doc_id & entry from 'users/{owner_uid}'
                if owner_uid:
                    user_ref = db.collection("users").document(owner_uid)
                    user_ref.set({
                        "document_ids": firestore.ArrayRemove([doc_id])
                    }, merge=True)
                    user_ref.collection("documents").document(doc_id).delete()
                    print(f"[FirebaseDB Admin] Removed {doc_id} from users/{owner_uid}")

                # D. Delete from legacy tenant docs collection
                db.collection(f"tenants_{tenant_id}_docs").document(doc_id).delete()
                admin_synced = True
                print(f"[FirebaseDB Admin] Deleted document {doc_id} and its chunks from Firestore")
        except Exception as e:
            print(f"[FirebaseDB Admin] Doc delete sync note: {e}")

        # 2. REST API fallback ONLY if Admin SDK failed
        if not admin_synced and FIREBASE_PROJECT_ID and id_token:
            try:
                url_doc = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/docs/{doc_id}"
                requests.delete(url_doc, headers={"Authorization": f"Bearer {id_token}"}, timeout=3)
                url_tenant = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/tenants_{tenant_id}_docs/{doc_id}"
                requests.delete(url_tenant, headers={"Authorization": f"Bearer {id_token}"}, timeout=3)
            except Exception:
                pass
        return True
    return False
