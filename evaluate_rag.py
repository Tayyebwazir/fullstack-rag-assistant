"""
RAG Evaluation Script: Measures Hit Rate@K and MRR on an Eval Set.

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --tenant db261c85-2d6f-4a93-8aeb-1f725a5e30cc --top-k 5
    python evaluate_rag.py --rerank
    python evaluate_rag.py --ingest
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH, TOP_K
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_documents, embed_query
from app.services.pdf_loader import load_pdf
from app.services.vector_store import (
    get_tenant_documents_summary,
    get_weaviate_tenants,
    ingest_chunks,
    search_chunks,
)

try:
    from app.services.reranker import rerank_chunks
except ImportError:
    rerank_chunks = None


# --- 10 Curated Ground-Truth Questions for 'Attention is all you need.pdf' ---
EVAL_SET = [
    {
        "id": 1,
        "question": "What is the mathematical equation for Scaled Dot-Product Attention?",
        "expected_pages": [4],
    },
    {
        "id": 2,
        "question": "How is Multi-Head Attention defined and how are multiple heads concatenated?",
        "expected_pages": [4, 5],
    },
    {
        "id": 3,
        "question": "What is the formula for the Position-wise Feed-Forward Networks (FFN)?",
        "expected_pages": [5],
    },
    {
        "id": 4,
        "question": "How are sinusoidal positional encodings calculated for even and odd dimensions?",
        "expected_pages": [6],
    },
    {
        "id": 5,
        "question": "According to Table 1, what is the maximum path length and per-layer complexity for self-attention?",
        "expected_pages": [6],
    },
    {
        "id": 6,
        "question": "What hardware and training schedule were used to train the base model?",
        "expected_pages": [7],
    },
    {
        "id": 7,
        "question": "What optimizer parameters and warmup steps schedule were used during training?",
        "expected_pages": [7],
    },
    {
        "id": 8,
        "question": "What residual dropout rate and label smoothing value were applied for regularization?",
        "expected_pages": [8],
    },
    {
        "id": 9,
        "question": "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?",
        "expected_pages": [8],
    },
    {
        "id": 10,
        "question": "How does the decoder prevent positions from attending to subsequent positions using masking?",
        "expected_pages": [3, 5],
    },
]


def resolve_tenant(requested_tenant: str | None) -> str:
    existing_tenants = get_weaviate_tenants()

    if requested_tenant:
        if requested_tenant not in existing_tenants:
            raise ValueError(
                f"Tenant '{requested_tenant}' not found in Weaviate"
            )
        return requested_tenant

    if len(existing_tenants) == 1:
        return existing_tenants[0]

    raise ValueError("Please specify a tenant using --tenant")


def ensure_document_ingested(tenant_id: str, pdf_file: Path) -> int:
    """Checks if the document already exists in the tenant; ingests it if missing."""
    summary = get_tenant_documents_summary(tenant_id=tenant_id)
    doc_counts = {item["source"]: item["chunk_count"] for item in summary}

    if pdf_file.name in doc_counts and doc_counts[pdf_file.name] > 0:
        print(f"[+] Found {doc_counts[pdf_file.name]} existing chunks for '{pdf_file.name}' in tenant '{tenant_id}'.")
        return doc_counts[pdf_file.name]

    print(f"[*] Ingesting '{pdf_file.name}' into tenant '{tenant_id}'...")
    pages = load_pdf(pdf_file)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    contents = [c["content"] for c in chunks]
    vectors = embed_documents(contents)
    total = ingest_chunks(
        chunks=chunks,
        source=pdf_file.name,
        vectors=vectors,
        tenant_id=tenant_id,
    )
    print(f"[+] Successfully ingested {total} chunks into tenant '{tenant_id}'.\n")
    return total


def run_evaluation(tenant_id: str, top_k: int = TOP_K, use_rerank: bool = False):
    """Executes retrieval evaluation and prints Hit Rank and Precision@K per question."""
    print(f"{'=' * 80}")
    print(f" RAG RETRIEVAL EVALUATION")
    print(f" Tenant: {tenant_id} | Top-K: {top_k} | Reranker: {'ON' if use_rerank else 'OFF'}")
    print(f"{'=' * 80}\n")

    for item in EVAL_SET:
        qid = item["id"]
        q = item["question"]
        expected = item["expected_pages"]

        # 1. Embed query & search vector database
        q_vec = embed_query(q)
        chunks = search_chunks(
            query_vector=q_vec,
            tenant_id=tenant_id,
            source_filter=PDF_PATH.name,
            limit=top_k,
        )

        # 2. Optionally rerank chunks
        if use_rerank and rerank_chunks and chunks:
            chunks = rerank_chunks(question=q, chunks=chunks, top_k=top_k)

        retrieved_pages = [c.get("page") for c in chunks]

        # 3. Hit rank & status
        hit_k = any(p in expected for p in retrieved_pages)
        first_hit_rank = "-"
        for rank, p in enumerate(retrieved_pages, start=1):
            if p in expected:
                first_hit_rank = str(rank)
                break

        # 4. Precision@K = (Correct retrieved chunks / Total retrieved chunks) * 100
        correct_chunks = sum(1 for p in retrieved_pages if p in expected)
        total_chunks = len(retrieved_pages)
        precision_k = (correct_chunks / total_chunks * 100) if total_chunks > 0 else 0.0

        # 5. Display result per query
        status = "PASS" if hit_k else "FAIL"
        print(f"[{status}] Q{qid:02d}: {q}")
        print(f"       Expected: {expected} | Retrieved: {retrieved_pages} | Hit Rank: {first_hit_rank} | Precision@{top_k}: {precision_k:.1f}% ({correct_chunks}/{total_chunks})")

    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Hit Rate")
    parser.add_argument("--tenant", type=str, default=None, help="Weaviate tenant ID (defaults to active tenant)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved chunks")
    parser.add_argument("--rerank", action="store_true", help="Enable Gemini reranking")
    parser.add_argument("--ingest", action="store_true", help="Force re-ingest PDF before evaluation")
    args = parser.parse_args()

    tenant = resolve_tenant(args.tenant)

    if args.ingest:
        pages = load_pdf(PDF_PATH)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        vectors = embed_documents([c["content"] for c in chunks])
        ingest_chunks(chunks=chunks, source=PDF_PATH.name, vectors=vectors, tenant_id=tenant)
    else:
        ensure_document_ingested(tenant_id=tenant, pdf_file=PDF_PATH)

    run_evaluation(tenant_id=tenant, top_k=args.top_k, use_rerank=args.rerank)


if __name__ == "__main__":
    main()
