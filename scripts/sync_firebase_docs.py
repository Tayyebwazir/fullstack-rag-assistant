import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.firebase_db import (
    _get_firestore_client,
    _resolve_user_email,
    _sync_user_document_entry,
    save_document_chunks,
)
from app.services.vector_store import connect_weaviate, COLLECTION_NAME
from weaviate.classes.query import Filter
from firebase_admin import firestore

DOCUMENTS_STORE_FILE = Path("app/data/firebase_documents.json")
USERS_STORE_FILE = Path("app/data/firebase_users.json")


def sync_all_to_firestore():
    db = _get_firestore_client()
    if not db:
        print("Failed to get Firestore client!")
        return

    print("Connected to Firestore. Starting sync...")

    with open(DOCUMENTS_STORE_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    local_users = {}
    if USERS_STORE_FILE.exists():
        with open(USERS_STORE_FILE, "r", encoding="utf-8") as f:
            local_users = json.load(f)

    # Ensure active user 7thay2TWAQRwQJRCEKwpaC0ZHj93 keeps its real Weaviate tenant db261c85-2d6f-4a93-8aeb-1f725a5e30cc
    active_uid = "7thay2TWAQRwQJRCEKwpaC0ZHj93"
    real_tenant = "db261c85-2d6f-4a93-8aeb-1f725a5e30cc"
    if active_uid in local_users:
        local_users[active_uid]["tenant_id"] = real_tenant

    # 1. Group documents by owner_uid
    user_docs_map: dict[str, list[str]] = {}
    user_uploaded_map: dict[str, dict[str, str]] = {}
    for doc_id, doc in documents.items():
        owner_uid = doc.get("owner_uid", "")
        owner_email = _resolve_user_email(owner_uid, doc.get("owner_email", ""))
        doc["owner_email"] = owner_email
        if owner_uid:
            user_docs_map.setdefault(owner_uid, []).append(doc_id)
            user_uploaded_map.setdefault(owner_uid, {})[doc_id] = doc.get("filename", "")

    # Save enriched owner_email back to local firebase_documents.json
    with open(DOCUMENTS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    # 2. Sync all documents into root collection 'docs' and legacy tenant collection
    for doc_id, doc in documents.items():
        tenant_id = doc.get("tenant_id", "default")
        clean_doc = {
            "doc_id": doc_id,
            "filename": doc.get("filename", ""),
            "file_type": doc.get("file_type", ""),
            "file_size": doc.get("file_size", 0),
            "tenant_id": tenant_id,
            "owner_uid": doc.get("owner_uid", ""),
            "owner_email": doc.get("owner_email", ""),
            "status": doc.get("status", "uploaded"),
            "chunk_count": doc.get("chunk_count", 0),
            "version": doc.get("version", 1),
            "upload_timestamp": doc.get("upload_timestamp", ""),
            "updated_timestamp": doc.get("updated_timestamp", ""),
            "error_message": doc.get("error_message"),
        }
        db.collection("docs").document(doc_id).set(clean_doc, merge=True)
        db.collection(f"tenants_{tenant_id}_docs").document(doc_id).set(clean_doc, merge=True)
        print(f"Synced doc {doc_id} ('{clean_doc['filename']}', owner={clean_doc['owner_email']}) to docs/{doc_id}")

    # 3. Sync EVERY Firestore user with exact document_ids and users/{uid}/documents subcollection (and remove uploaded_documents)
    for user_snap in db.collection("users").stream():
        uid = user_snap.id
        doc_ids = user_docs_map.get(uid, [])

        user_patch = {
            "document_ids": doc_ids,
        }
        if uid == active_uid:
            user_patch["tenant_id"] = real_tenant

        db.collection("users").document(uid).set(user_patch, merge=True)
        try:
            db.collection("users").document(uid).update({
                "uploaded_documents": firestore.DELETE_FIELD
            })
        except Exception:
            pass

        # Sync subcollection users/{uid}/documents/{doc_id}
        docs_subcoll = db.collection("users").document(uid).collection("documents")
        for old_ref in docs_subcoll.stream():
            if old_ref.id not in doc_ids:
                old_ref.reference.delete()

        for doc_id in doc_ids:
            d = documents[doc_id]
            summary = {
                "doc_id": doc_id,
                "filename": d.get("filename", ""),
                "file_type": d.get("file_type", ""),
                "status": d.get("status", ""),
                "chunk_count": d.get("chunk_count", 0),
                "tenant_id": d.get("tenant_id", ""),
                "upload_timestamp": d.get("upload_timestamp", ""),
                "updated_timestamp": d.get("updated_timestamp", ""),
            }
            _sync_user_document_entry(db, uid, doc_id, summary)

        print(f"Updated users/{uid}: {len(doc_ids)} docs (uploaded_documents removed)")

    # Also update local firebase_users.json to match Firestore cleanly
    for uid, u_data in local_users.items():
        u_data["document_ids"] = user_docs_map.get(uid, [])
        u_data.pop("uploaded_documents", None)
    with open(USERS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(local_users, f, indent=2)

    # 4. Sync chunks from Weaviate into docs/{doc_id}/chunks if missing
    try:
        with connect_weaviate() as w_client:
            coll = w_client.collections.get(COLLECTION_NAME)
            for doc_id, doc in documents.items():
                tenant_id = doc.get("tenant_id")
                if not tenant_id or doc.get("status") != "completed":
                    continue
                try:
                    tenant = coll.with_tenant(tenant_id)
                    res = tenant.query.fetch_objects(
                        filters=Filter.by_property("doc_id").equal(doc_id),
                        limit=200
                    )
                    if res.objects:
                        sorted_objects = sorted(res.objects, key=lambda o: o.properties.get("chunk_index", 0))
                        chunks = [
                            {
                                "chunk_index": obj.properties.get("chunk_index", 0),
                                "content": obj.properties.get("content", ""),
                                "page": obj.properties.get("page", 1),
                                "source": obj.properties.get("source", doc.get("filename", "")),
                                "doc_id": doc_id,
                                "tenant_id": tenant_id,
                            }
                            for obj in sorted_objects
                        ]
                        save_document_chunks(
                            doc_id=doc_id,
                            chunks=chunks,
                            tenant_id=tenant_id,
                            source=doc.get("filename", ""),
                        )
                except Exception as e:
                    print(f"Note querying Weaviate chunks for doc {doc_id}: {e}")
    except Exception as e:
        print(f"Weaviate connection note: {e}")

    print("\n--- Sync Complete ---")


if __name__ == "__main__":
    sync_all_to_firestore()
