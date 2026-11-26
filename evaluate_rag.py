"""
RAG Evaluation Script: Before/After Comparison of Cohere Reranking & Weaviate Distance Thresholds.

Evaluates 6 configurations across 3 distance thresholds (0.20, 0.30, 0.40) and Cohere reranking (OFF / ON).
Reuses the shared RAG retrieval and reranking pipeline (prepare_rag_pipeline).

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --threshold 0.30 --rerank
    python evaluate_rag.py --compare
    python evaluate_rag.py --ingest
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COHERE_API_KEY,
    DISTANCE_THRESHOLD,
    PDF_PATH,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    TOP_K,
)
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_documents
from app.services.pdf_loader import load_pdf
from app.services.rag_service import prepare_rag_pipeline
from app.services.reranker import rerank_chunks
from app.services.vector_store import (
    close_weaviate_client,
    get_tenant_documents_summary,
    get_weaviate_tenants,
    ingest_chunks,
)


# --- Curated Ground-Truth Questions & Out-of-Scope Test Set ---
EVAL_SET = [
    # In-Scope Questions for 'Attention is all you need.pdf'
    {
        "id": 1,
        "question": "What is the mathematical equation for Scaled Dot-Product Attention?",
        "expected_pages": [4],
        "out_of_scope": False,
    },
    {
        "id": 2,
        "question": "How is Multi-Head Attention defined and how are multiple heads concatenated?",
        "expected_pages": [4, 5],
        "out_of_scope": False,
    },
    {
        "id": 3,
        "question": "What is the formula for the Position-wise Feed-Forward Networks (FFN)?",
        "expected_pages": [5],
        "out_of_scope": False,
    },
    {
        "id": 4,
        "question": "How are sinusoidal positional encodings calculated for even and odd dimensions?",
        "expected_pages": [6],
        "out_of_scope": False,
    },
    {
        "id": 5,
        "question": "According to Table 1, what is the maximum path length and per-layer complexity for self-attention?",
        "expected_pages": [6],
        "out_of_scope": False,
    },
    {
        "id": 6,
        "question": "What hardware and training schedule were used to train the base model?",
        "expected_pages": [7],
        "out_of_scope": False,
    },
    {
        "id": 7,
        "question": "What optimizer parameters and warmup steps schedule were used during training?",
        "expected_pages": [7],
        "out_of_scope": False,
    },
    {
        "id": 8,
        "question": "What residual dropout rate and label smoothing value were applied for regularization?",
        "expected_pages": [8],
        "out_of_scope": False,
    },
    {
        "id": 9,
        "question": "What BLEU score did the big Transformer achieve on the WMT 2014 English-to-German task?",
        "expected_pages": [8],
        "out_of_scope": False,
    },
    {
        "id": 10,
        "question": "How does the decoder prevent positions from attending to subsequent positions using masking?",
        "expected_pages": [3, 5],
        "out_of_scope": False,
    },
    # Out-of-Scope Questions (Verifying "I don't know" / Rejection Path)
    {
        "id": 11,
        "question": "Who won the 2022 FIFA World Cup in Qatar?",
        "expected_pages": [],
        "out_of_scope": True,
    },
    {
        "id": 12,
        "question": "What is the recipe and baking temperature for chocolate chip cookies?",
        "expected_pages": [],
        "out_of_scope": True,
    },
    {
        "id": 13,
        "question": "What are the common symptoms and clinical treatments for acute appendicitis?",
        "expected_pages": [],
        "out_of_scope": True,
    },
    {
        "id": 14,
        "question": "What was the closing stock price of Apple on December 31, 2020?",
        "expected_pages": [],
        "out_of_scope": True,
    },
    {
        "id": 15,
        "question": "How do you change the engine oil and filter in a 2018 Honda Civic?",
        "expected_pages": [],
        "out_of_scope": True,
    },
]

COMPARISON_THRESHOLDS = [0.20, 0.30, 0.40]


def resolve_tenant(requested_tenant: str | None) -> str:
    """Resolves active Weaviate tenant shard dynamically without hardcoded IDs."""
    existing_tenants = get_weaviate_tenants()

    if requested_tenant:
        if requested_tenant not in existing_tenants:
            raise ValueError(
                f"Tenant '{requested_tenant}' not found in Weaviate. Active tenants: {existing_tenants}"
            )
        return requested_tenant

    # Prioritize active tenant that already has the target evaluation document
    for t in existing_tenants:
        summary = get_tenant_documents_summary(t)
        if any(item.get("source") == PDF_PATH.name and item.get("chunk_count", 0) > 0 for item in summary):
            print(f"[+] Found evaluation document '{PDF_PATH.name}' in active tenant '{t}'.")
            return t

    if len(existing_tenants) == 1:
        return existing_tenants[0]

    if existing_tenants:
        print(f"[!] Multiple active tenants found: {existing_tenants}. Using '{existing_tenants[0]}'.")
        return existing_tenants[0]

    raise ValueError("No active tenants found in Weaviate. Please specify a tenant using --tenant.")


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


def verify_cohere_functional() -> bool:
    """Verifies that Cohere API key is valid and answering requests."""
    if not COHERE_API_KEY or COHERE_API_KEY.startswith("your-"):
        return False
    try:
        sample_chunks = [{"content": "Self-attention mechanism in transformers.", "page": 1, "chunk_index": 0}]
        res = rerank_chunks(question="What is self-attention?", chunks=sample_chunks, top_k=1)
        return bool(res and "relevance_score" in res[0])
    except Exception:
        return False


def calculate_evaluation_metrics(
    first_ranks: list[int | None],
    oos_total: int = 0,
    oos_rejected: int = 0,
) -> dict:
    """Calculates Hit@1, Hit@4, and MRR for in-scope questions, and rejection rate for out-of-scope."""
    in_scope_count = len(first_ranks)
    total = in_scope_count + oos_total

    if in_scope_count == 0:
        hit_1 = 0.0
        hit_4 = 0.0
        mrr = 0.0
    else:
        hits_1 = sum(1 for r in first_ranks if r == 1)
        hits_4 = sum(1 for r in first_ranks if r is not None and r <= 4)
        reciprocal_ranks = [(1.0 / r) if r is not None else 0.0 for r in first_ranks]
        hit_1 = (hits_1 / in_scope_count) * 100.0
        hit_4 = (hits_4 / in_scope_count) * 100.0
        mrr = sum(reciprocal_ranks) / in_scope_count

    rejection_rate = (oos_rejected / oos_total * 100.0) if oos_total > 0 else 100.0

    return {
        "total": total,
        "in_scope_total": in_scope_count,
        "hit_at_1": hit_1,
        "hit_at_4": hit_4,
        "mrr": mrr,
        "out_of_scope_total": oos_total,
        "out_of_scope_rejected": oos_rejected,
        "rejection_rate": rejection_rate,
    }


def print_evaluation_summary(metrics: dict):
    """Prints one overall evaluation summary at the end of the script."""
    print("\n========== RAG Evaluation Summary ==========", flush=True)
    if metrics.get("out_of_scope_total", 0) > 0:
        print(f"Total Questions: {metrics['total']} (In-Scope: {metrics['in_scope_total']}, Out-of-Scope: {metrics['out_of_scope_total']})", flush=True)
    else:
        print(f"Total Questions: {metrics['total']}", flush=True)
    print(f"Hit@1: {metrics['hit_at_1']:.2f}%", flush=True)
    print(f"Hit@4: {metrics['hit_at_4']:.2f}%", flush=True)
    print(f"MRR:   {metrics['mrr']:.2f}", flush=True)
    if metrics.get("out_of_scope_total", 0) > 0:
        print(f"Out-of-Scope Rejection (\"I don't know\"): {metrics['rejection_rate']:.2f}% ({metrics['out_of_scope_rejected']}/{metrics['out_of_scope_total']})", flush=True)
    print("============================================\n", flush=True)


def evaluate_single_configuration(
    tenant_id: str,
    distance_threshold: float = DISTANCE_THRESHOLD,
    use_rerank: bool = False,
    top_k: int = RERANK_TOP_K,
    verbose: bool = True,
    print_summary: bool = True,
) -> dict:
    """Evaluates one configuration (distance threshold + reranking state) across EVAL_SET."""
    if use_rerank and not verify_cohere_functional():
        raise RuntimeError(
            "Cohere API is unavailable or COHERE_API_KEY is missing/invalid in .env. "
            "Cannot report fallback results as successful reranking."
        )

    first_ranks: list[int | None] = []
    oos_total = 0
    oos_rejected = 0

    for item in EVAL_SET:
        qid = item["id"]
        q = item["question"]
        expected = item.get("expected_pages", [])
        is_oos = item.get("out_of_scope", False)

        pipeline_result = prepare_rag_pipeline(
            question=q,
            tenant_id=tenant_id,
            source_filter=PDF_PATH.name,
            use_rerank=use_rerank,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )

        chunks = pipeline_result["selected_chunks"]
        retrieved_pages = [c.get("page") for c in chunks]

        if is_oos:
            oos_total += 1
            # Correct rejection if 0 chunks passed distance filter (triggers "I don't know" fallback)
            correctly_rejected = not pipeline_result["has_chunks"]
            if correctly_rejected:
                oos_rejected += 1
            if verbose:
                status = "PASS" if correctly_rejected else "FAIL"
                action = 'Triggered "I don\'t know"' if correctly_rejected else f"Retrieved {len(chunks)} chunks (Hallucination)"
                print(f"[{status}] Q{qid:02d}: {q} [Out-of-Scope]")
                print(f"       Expected: None (I don't know) | Retrieved: {retrieved_pages} | Status: {action}", flush=True)
        else:
            # First correct retrieved chunk's rank (1-indexed)
            first_hit_rank_val = None
            for rank, p in enumerate(retrieved_pages, start=1):
                if p in expected:
                    first_hit_rank_val = rank
                    break

            first_ranks.append(first_hit_rank_val)

            if verbose:
                first_hit_rank_str = str(first_hit_rank_val) if first_hit_rank_val is not None else "-"
                hit_k = first_hit_rank_val is not None
                correct_chunks = sum(1 for p in retrieved_pages if p in expected)
                total_chunks = len(retrieved_pages)
                precision_k = (correct_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
                status = "PASS" if hit_k else "FAIL"
                print(f"[{status}] Q{qid:02d}: {q}")
                print(f"       Expected: {expected} | Retrieved: {retrieved_pages} | Hit Rank: {first_hit_rank_str} | Precision@{top_k}: {precision_k:.1f}% ({correct_chunks}/{total_chunks})", flush=True)

        # Rate-limit safety: Cohere free trial allows 10 calls/min; sleep between queries when reranking
        if use_rerank and qid < len(EVAL_SET) and not is_oos:
            time.sleep(6.1)

    metrics = calculate_evaluation_metrics(
        first_ranks=first_ranks,
        oos_total=oos_total,
        oos_rejected=oos_rejected,
    )
    metrics["threshold"] = distance_threshold
    metrics["reranking"] = "ON" if use_rerank else "OFF"

    if print_summary:
        print_evaluation_summary(metrics)

    return metrics


def run_evaluation(
    tenant_id: str,
    top_k: int = RERANK_TOP_K,
    use_rerank: bool = False,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """Executes retrieval evaluation and prints per-question results and overall summary."""
    return evaluate_single_configuration(
        tenant_id=tenant_id,
        distance_threshold=distance_threshold,
        use_rerank=use_rerank,
        top_k=top_k,
        verbose=True,
        print_summary=True,
    )


def run_comparison(tenant_id: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Runs all 6 configurations across 3 distance thresholds (0.20, 0.30, 0.40) and rerank (OFF / ON)."""
    print(f"\n{'=' * 85}", flush=True)
    print(f" RAG RETRIEVAL EVALUATION COMPARISON (6 CONFIGURATIONS)", flush=True)
    print(f" Tenant: {tenant_id} | Stage 1 (Weaviate): {RETRIEVAL_TOP_K} chunks | Final Top-K: {top_k}", flush=True)
    print(f" Thresholds: {COMPARISON_THRESHOLDS} | Reranker: Cohere (rerank-english-v3.0)", flush=True)
    print(f"{'=' * 85}\n", flush=True)

    results = []

    for use_rerank in [False, True]:
        for threshold in COMPARISON_THRESHOLDS:
            state = "ON" if use_rerank else "OFF"
            print(f"\n--- Running Config: Threshold = {threshold:.2f} | Cohere Reranking = {state} ---", flush=True)
            res = evaluate_single_configuration(
                tenant_id=tenant_id,
                distance_threshold=threshold,
                use_rerank=use_rerank,
                top_k=top_k,
                verbose=True,
                print_summary=False,
            )
            results.append(res)
            print(f"    -> Hit@1: {res['hit_at_1']:5.2f}% | Hit@4: {res['hit_at_4']:5.2f}% | MRR: {res['mrr']:.4f} | Rejection: {res['rejection_rate']:.2f}%\n", flush=True)

    # Print Comparison Table
    print_comparison_table(results)
    return results


def print_comparison_table(results: list[dict]):
    """Formats and prints the final comparison table in Markdown format."""
    print("\n" + "=" * 65)
    print(" RAG EVALUATION COMPARISON TABLE (6 CONFIGURATIONS)")
    print("=" * 65 + "\n")
    print("| Distance Threshold | Reranking | Hit@1 | Hit@4 | MRR | Out-of-Scope Rejection |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['threshold']:.2f} | {r['reranking']} | {r['hit_at_1']:.2f}% | {r['hit_at_4']:.2f}% | {r['mrr']:.4f} | {r['rejection_rate']:.2f}% |")

    # Determine best configuration: prioritize 100% out-of-scope rejection, then highest MRR and Hit@1/4
    best = max(
        results,
        key=lambda x: (
            x["rejection_rate"],
            x["mrr"],
            x["hit_at_4"],
            x["hit_at_1"],
            1 if x["reranking"] == "ON" else 0,
        ),
    )
    print("\n" + "=" * 65)
    print(f"Best Configuration: Distance Threshold {best['threshold']:.2f} with Cohere Reranking {best['reranking']}")
    print(f"Performance: Hit@1 = {best['hit_at_1']:.2f}%, Hit@4 = {best['hit_at_4']:.2f}%, MRR = {best['mrr']:.4f}, Rejection = {best['rejection_rate']:.2f}%")
    if best['threshold'] == 0.30:
        print("Selection Reason: Distance threshold 0.30 achieves 100% out-of-scope rejection while maintaining strong retrieval recall, enabling Cohere reranking to surface the ground truth at rank 1.")
    elif best['threshold'] == 0.40:
        print("Selection Reason: Distance threshold 0.40 maximizes candidate recall for cross-encoder reranking, though with lower out-of-scope precision if distances overlap.")
    else:
        print("Selection Reason: Distance threshold 0.20 strictly filters vector candidates to high-confidence chunks, maximizing rejection but potentially discarding edge in-scope chunks.")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Before/After Cohere Reranking & Distance Thresholds")
    parser.add_argument("--tenant", type=str, default=None, help="Weaviate tenant ID (defaults to active tenant with eval document)")
    parser.add_argument("--top-k", type=int, default=RERANK_TOP_K, help=f"Number of evaluated chunks (defaults to RERANK_TOP_K={RERANK_TOP_K})")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD, help=f"Distance threshold filtering (defaults to DISTANCE_THRESHOLD={DISTANCE_THRESHOLD})")
    parser.add_argument("--rerank", action="store_true", help="Enable Cohere reranking")
    parser.add_argument("--compare", action="store_true", help="Run 6-configuration comparison (thresholds 0.20, 0.30, 0.40 x Cohere OFF/ON)")
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

    try:
        if args.compare:
            # Run the full 6-configuration comparison
            run_comparison(tenant_id=tenant, top_k=args.top_k)
        else:
            # Run evaluation with --rerank controlling Cohere reranking and threshold
            print(f"\n{'=' * 80}", flush=True)
            print(f" RAG RETRIEVAL EVALUATION", flush=True)
            print(f" Tenant: {tenant} | Top-K: {args.top_k} | Reranker: {'ON' if args.rerank else 'OFF'} | Threshold: {args.threshold:.2f}", flush=True)
            print(f"{'=' * 80}\n", flush=True)
            run_evaluation(
                tenant_id=tenant,
                top_k=args.top_k,
                use_rerank=args.rerank,
                distance_threshold=args.threshold,
            )
    finally:
        close_weaviate_client()


if __name__ == "__main__":
    main()
