import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.firebase_db import _get_firestore_client

KEEP_IDS = {
    "608ad9a4-f144-4c3c-836c-84e27923eae5",
    "f4ce8a57-9a66-40a2-b0a3-d0524e980deb",
    "5690a748-be51-40f7-afc7-36a434cc2733",
}

DOCS_JSON_PATH = Path("app/data/firebase_documents.json")
USERS_JSON_PATH = Path("app/data/firebase_users.json")
ACTIVE_UID = "7thay2TWAQRwQJRCEKwpaC0ZHj93"

def cleanup():
    db = _get_firestore_client()
    if not db:
        print("Could not connect to Firestore")
        return

    print("Cleaning up Firestore docs collection...")
    # 1. Delete all docs from Firestore 'docs' except the 3 KEEP_IDS
    docs_ref = db.collection("docs")
    for doc in docs_ref.stream():
        if doc.id not in KEEP_IDS:
            # Delete any subcollection chunks
            for chunk in doc.reference.collection("chunks").stream():
                chunk.reference.delete()
            doc.reference.delete()
            print(f"Deleted old doc from Firestore: {doc.id}")

    # 2. Update user's document_ids array in Firestore
    user_ref = db.collection("users").document(ACTIVE_UID)
    keep_list = [
        "608ad9a4-f144-4c3c-836c-84e27923eae5",
        "f4ce8a57-9a66-40a2-b0a3-d0524e980deb",
        "5690a748-be51-40f7-afc7-36a434cc2733",
    ]
    user_ref.set({"document_ids": keep_list}, merge=True)
    print(f"Updated user {ACTIVE_UID}.document_ids to strictly: {keep_list}")

    # 3. Clean up other users' document_ids
    for user_doc in db.collection("users").stream():
        if user_doc.id != ACTIVE_UID:
            user_doc.reference.set({"document_ids": []}, merge=True)

    # 4. Clean up local json files
    if DOCS_JSON_PATH.exists():
        with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
            all_docs = json.load(f)
        filtered_docs = {k: v for k, v in all_docs.items() if k in KEEP_IDS}
        with open(DOCS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_docs, f, indent=2)
        print(f"Cleaned up {DOCS_JSON_PATH.name}: kept {len(filtered_docs)} documents")

    if USERS_JSON_PATH.exists():
        with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
            all_users = json.load(f)
        if ACTIVE_UID in all_users:
            all_users[ACTIVE_UID]["document_ids"] = keep_list
        with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_users, f, indent=2)
        print(f"Cleaned up {USERS_JSON_PATH.name}")

    print("\nCleanup successfully completed!")

if __name__ == "__main__":
    cleanup()
