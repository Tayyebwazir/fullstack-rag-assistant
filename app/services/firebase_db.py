import json
from pathlib import Path
import requests
from app.config import FIREBASE_PROJECT_ID, BASE_DIR

USERS_STORE_FILE = BASE_DIR / "app" / "data" / "firebase_users.json"

DEFAULT_PERMISSIONS = ["read", "write", "update", "delete", "ingest"]
VALID_TENANTS = ["alpha", "beta", "tri"]


def _load_local_store() -> dict:
    if USERS_STORE_FILE.exists():
        try:
            with open(USERS_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_local_store(data: dict):
    USERS_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_user_record(uid: str, email: str = "", id_token: str | None = None) -> dict:
    """Fetches user UUID, tenant ID, and permissions from Firebase Database.
    Falls back to persistent local storage for high availability.
    """
    local_data = _load_local_store()

    # 1. Check local persistent store first
    if uid in local_data:
        record = local_data[uid]
        if email and not record.get("email"):
            record["email"] = email
            _save_local_store(local_data)
        return record

    # 2. Try fetching from Firebase Firestore REST API
    firestore_record = None
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            headers = {"Authorization": f"Bearer {id_token}"}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                fields = resp.json().get("fields", {})
                tenant = fields.get("tenant_id", {}).get("stringValue", "alpha")
                perms_array = fields.get("permissions", {}).get("arrayValue", {}).get("values", [])
                perms = [p.get("stringValue") for p in perms_array if p.get("stringValue")] or DEFAULT_PERMISSIONS
                firestore_record = {
                    "uid": uid,
                    "email": email,
                    "tenant_id": tenant,
                    "permissions": perms,
                }
        except Exception as e:
            print(f"[FirebaseDB] Firestore lookup note: {e}")

    if firestore_record:
        local_data[uid] = firestore_record
        _save_local_store(local_data)
        return firestore_record

    # 3. New user registration: default to tenant 'alpha' with full permissions
    new_record = {
        "uid": uid,
        "email": email,
        "tenant_id": "alpha",
        "permissions": DEFAULT_PERMISSIONS,
    }
    local_data[uid] = new_record
    _save_local_store(local_data)

    # Optionally sync back to Firestore
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            payload = {
                "fields": {
                    "uid": {"stringValue": uid},
                    "email": {"stringValue": email or ""},
                    "tenant_id": {"stringValue": new_record["tenant_id"]},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in DEFAULT_PERMISSIONS]}},
                }
            }
            requests.patch(url, headers={"Authorization": f"Bearer {id_token}"}, json=payload, timeout=3)
        except Exception:
            pass

    return new_record


def update_user_tenant_and_permissions(
    uid: str,
    tenant_id: str,
    permissions: list[str] | None = None,
    id_token: str | None = None,
) -> dict:
    """Updates the user's tenant assignment and permissions in Firebase Database."""
    if tenant_id not in VALID_TENANTS:
        raise ValueError(f"Invalid tenant '{tenant_id}'. Must be one of: {VALID_TENANTS}")

    perms = permissions if permissions is not None else DEFAULT_PERMISSIONS
    local_data = _load_local_store()
    current = local_data.get(uid, {"uid": uid, "email": ""})
    current["tenant_id"] = tenant_id
    current["permissions"] = perms
    local_data[uid] = current
    _save_local_store(local_data)

    # Sync to Firestore if id_token provided
    if FIREBASE_PROJECT_ID and id_token:
        try:
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{uid}"
            payload = {
                "fields": {
                    "uid": {"stringValue": uid},
                    "tenant_id": {"stringValue": tenant_id},
                    "permissions": {"arrayValue": {"values": [{"stringValue": p} for p in perms]}},
                }
            }
            requests.patch(url, headers={"Authorization": f"Bearer {id_token}"}, json=payload, timeout=3)
        except Exception:
            pass

    return current


def verify_tenant_access(user_record: dict, target_tenant: str, required_permission: str) -> tuple[bool, str]:
    """Enforces:
    1. User belongs to requested tenant (or is authorized for it).
    2. User possesses the required action permission ('read', 'write', 'update', 'delete', 'ingest').
    """
    user_tenant = user_record.get("tenant_id", "alpha")
    user_perms = user_record.get("permissions", [])

    # 1. Tenant Check
    if target_tenant != user_tenant:
        return (
            False,
            f"Tenant Authorisation Failed: User belongs to tenant '{user_tenant}', but requested operation on tenant '{target_tenant}'.",
        )

    # 2. Permission Check
    if required_permission not in user_perms:
        return (
            False,
            f"Permission Denied: User lacks the '{required_permission}' permission on tenant '{target_tenant}'. Required: {required_permission}.",
        )

    return (True, "")
