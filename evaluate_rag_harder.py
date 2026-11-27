"""
RAG Harder Evaluation Script — Feature: feature/harder-evaluation

Improvements over the baseline evaluate_rag.py:
  - Loads questions from eval_dataset_harder.json (25 in-scope + 7 out-of-scope = 32 total)
  - Question types: exact_term, paraphrase, multi_hop, multi_page, out_of_scope
  - Hit detection: document name + page hit + answer-content keyword match
  - Multi-hop/multi-page: measures full evidence retrieval (all required pages found)
  - Tracks avg retrieval latency per configuration
  - Removes hardcoded threshold-selection reason strings; derives recommendation from results
  - New --harder flag to switch between baseline and harder dataset
  - New --dataset flag to load a custom JSON dataset file

Usage:
    python evaluate_rag_harder.py                            # harder dataset, single default config
    python evaluate_rag_harder.py --compare                  # 6-config comparison, harder dataset
    python evaluate_rag_harder.py --dataset eval_dataset_harder.json --compare
    python evaluate_rag_harder.py --threshold 0.30 --rerank  # single config with reranking
    python evaluate_rag_harder.py --ingest                   # force re-ingest then evaluate
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COHERE_API_KEY,
    DISTANCE_THRESHOLD,
    HYBRID_ALPHA,
    HYBRID_SCORE_THRESHOLD,
    PDF_PATH,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    SEARCH_MODE,
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

# ---------------------------------------------------------------------------
# Default dataset path
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH = BASE_DIR / "eval_dataset_harder.json"
COMPARISON_THRESHOLDS = [0.20, 0.30, 0.40]
HYBRID_ALPHAS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path) -> list[dict]:
    """Loads evaluation questions from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}. "
            "Run the script from the project root or specify --dataset."
        )
    with open(path, encoding="utf-8") as fh:
        dataset = json.load(fh)
    # Validate structure
    for item in dataset:
        assert "id" in item and "question" in item and "out_of_scope" in item, (
            f"Dataset item missing required fields: {item}"
        )
    return dataset


# ---------------------------------------------------------------------------
# Tenant Resolution
# ---------------------------------------------------------------------------

def resolve_tenant(requested_tenant: str | None) -> str:
    """Resolves active Weaviate tenant shard dynamically without hardcoded IDs."""
    existing_tenants = get_weaviate_tenants()

    if requested_tenant:
        if requested_tenant not in existing_tenants:
            raise ValueError(
                f"Tenant '{requested_tenant}' not found in Weaviate. "
                f"Active tenants: {existing_tenants}"
            )
        return requested_tenant

    for t in existing_tenants:
        summary = get_tenant_documents_summary(t)
        if any(
            item.get("source") == PDF_PATH.name and item.get("chunk_count", 0) > 0
            for item in summary
        ):
            print(f"[+] Found evaluation document '{PDF_PATH.name}' in tenant '{t}'.")
            return t

    if len(existing_tenants) == 1:
        return existing_tenants[0]
    if existing_tenants:
        print(f"[!] Multiple tenants found: {existing_tenants}. Using '{existing_tenants[0]}'.")
        return existing_tenants[0]

    raise ValueError("No active tenants found in Weaviate. Use --tenant to specify one.")


def ensure_document_ingested(tenant_id: str, pdf_file: Path) -> int:
    """Checks if the document exists; ingests it if missing."""
    summary = get_tenant_documents_summary(tenant_id=tenant_id)
    doc_counts = {item["source"]: item["chunk_count"] for item in summary}

    if pdf_file.name in doc_counts and doc_counts[pdf_file.name] > 0:
        print(
            f"[+] Found {doc_counts[pdf_file.name]} existing chunks "
            f"for '{pdf_file.name}' in tenant '{tenant_id}'."
        )
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
    print(f"[+] Ingested {total} chunks into tenant '{tenant_id}'.\n")
    return total


# ---------------------------------------------------------------------------
# Cohere Verification
# ---------------------------------------------------------------------------

def verify_cohere_functional() -> bool:
    """Verifies that Cohere API key is valid and responding."""
    if not COHERE_API_KEY or COHERE_API_KEY.startswith("your-"):
        return False
    try:
        sample = [{"content": "Self-attention mechanism in transformers.", "page": 1, "chunk_index": 0}]
        res = rerank_chunks(question="What is self-attention?", chunks=sample, top_k=1)
        return bool(res and "relevance_score" in res[0])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hit / Evidence Detection
# ---------------------------------------------------------------------------

def _keyword_hit(content: str, keywords: list[str]) -> bool:
    """Returns True if at least one required keyword appears in the chunk content."""
    if not keywords:
        return True
    text = content.lower()
    return any(kw.lower() in text for kw in keywords)


def _doc_name_hit(chunk: dict, expected_doc: str | None) -> bool:
    """Returns True if the chunk's source document matches the expected document."""
    if not expected_doc:
        return True
    return chunk.get("source", "").lower() == expected_doc.lower()


def _page_hit(chunk_page: int | None, expected_pages: list[int]) -> bool:
    """Returns True if chunk page is in expected pages list."""
    if not expected_pages:
        return False
    return chunk_page in expected_pages


def compute_hit_rank(
    chunks: list[dict],
    expected_pages: list[int],
    expected_doc: str | None,
    required_keywords: list[str],
) -> int | None:
    """
    Returns 1-indexed rank of first chunk that satisfies:
      - page in expected_pages
      - source doc matches expected_doc
      - at least one required keyword found in content (if any)
    Returns None if no such chunk found.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if not _page_hit(chunk.get("page"), expected_pages):
            continue
        if not _doc_name_hit(chunk, expected_doc):
            continue
        if required_keywords and not _keyword_hit(chunk.get("content", ""), required_keywords):
            continue
        return rank
    return None


def compute_evidence_coverage(
    chunks: list[dict],
    expected_pages: list[int],
    expected_doc: str | None,
) -> tuple[int, int]:
    """
    For multi-hop/multi-page questions, counts how many required pages were retrieved.
    Returns (pages_found, pages_required).
    """
    if not expected_pages:
        return 0, 0
    retrieved_pages = set()
    for chunk in chunks:
        if _doc_name_hit(chunk, expected_doc) and chunk.get("page") in expected_pages:
            retrieved_pages.add(chunk.get("page"))
    return len(retrieved_pages), len(expected_pages)


# ---------------------------------------------------------------------------
# Metric Calculation
# ---------------------------------------------------------------------------

def calculate_evaluation_metrics(
    first_ranks: list[int | None],
    oos_total: int = 0,
    oos_rejected: int = 0,
    keyword_hits: list[bool] | None = None,
    evidence_coverages: list[tuple[int, int]] | None = None,
    latencies_ms: list[float] | None = None,
) -> dict:
    """
    Calculates Hit@1, Hit@4, MRR, keyword-validated hits, complete evidence rate,
    out-of-scope rejection rate, and avg latency.
    """
    in_scope_count = len(first_ranks)
    total = in_scope_count + oos_total

    if in_scope_count == 0:
        hit_1 = hit_4 = mrr = 0.0
    else:
        hits_1 = sum(1 for r in first_ranks if r == 1)
        hits_4 = sum(1 for r in first_ranks if r is not None and r <= 4)
        reciprocal_ranks = [(1.0 / r) if r is not None else 0.0 for r in first_ranks]
        hit_1 = (hits_1 / in_scope_count) * 100.0
        hit_4 = (hits_4 / in_scope_count) * 100.0
        mrr = sum(reciprocal_ranks) / in_scope_count

    rejection_rate = (oos_rejected / oos_total * 100.0) if oos_total > 0 else 100.0

    # Keyword-validated hit@4 (content correctness, not just page)
    kw_hit_4 = None
    if keyword_hits is not None and in_scope_count > 0:
        kw_hit_4 = (sum(keyword_hits) / in_scope_count) * 100.0

    # Complete evidence retrieval rate (all required pages found)
    complete_evidence_rate = None
    if evidence_coverages is not None and in_scope_count > 0:
        complete_count = sum(
            1 for found, required in evidence_coverages
            if required > 0 and found >= required
        )
        multi_hop_count = sum(1 for _, required in evidence_coverages if required > 0)
        if multi_hop_count > 0:
            complete_evidence_rate = (complete_count / multi_hop_count) * 100.0

    avg_latency_ms = None
    if latencies_ms:
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

    return {
        "total": total,
        "in_scope_total": in_scope_count,
        "hit_at_1": hit_1,
        "hit_at_4": hit_4,
        "mrr": mrr,
        "out_of_scope_total": oos_total,
        "out_of_scope_rejected": oos_rejected,
        "rejection_rate": rejection_rate,
        "keyword_validated_hit_at_4": kw_hit_4,
        "complete_evidence_rate": complete_evidence_rate,
        "avg_latency_ms": avg_latency_ms,
    }


# ---------------------------------------------------------------------------
# Summary Printing
# ---------------------------------------------------------------------------

def print_evaluation_summary(metrics: dict):
    """Prints evaluation summary with all harder metrics."""
    print("\n========== RAG Evaluation Summary ==========", flush=True)
    if metrics.get("out_of_scope_total", 0) > 0:
        print(
            f"Total Questions : {metrics['total']} "
            f"(In-Scope: {metrics['in_scope_total']}, "
            f"Out-of-Scope: {metrics['out_of_scope_total']})",
            flush=True,
        )
    else:
        print(f"Total Questions : {metrics['total']}", flush=True)
    print(f"Hit@1           : {metrics['hit_at_1']:.2f}%", flush=True)
    print(f"Hit@4           : {metrics['hit_at_4']:.2f}%", flush=True)
    print(f"MRR             : {metrics['mrr']:.4f}", flush=True)
    if metrics.get("keyword_validated_hit_at_4") is not None:
        print(f"KW-Validated @4 : {metrics['keyword_validated_hit_at_4']:.2f}%", flush=True)
    if metrics.get("complete_evidence_rate") is not None:
        print(f"Full Evidence   : {metrics['complete_evidence_rate']:.2f}%", flush=True)
    if metrics.get("avg_latency_ms") is not None:
        print(f"Avg Latency     : {metrics['avg_latency_ms']:.1f} ms", flush=True)
    if metrics.get("out_of_scope_total", 0) > 0:
        print(
            f"OOS Rejection   : {metrics['rejection_rate']:.2f}% "
            f"({metrics['out_of_scope_rejected']}/{metrics['out_of_scope_total']})",
            flush=True,
        )
    print("============================================\n", flush=True)


# ---------------------------------------------------------------------------
# Single Configuration Evaluator
# ---------------------------------------------------------------------------

def evaluate_single_configuration(
    tenant_id: str,
    eval_set: list[dict],
    distance_threshold: float = DISTANCE_THRESHOLD,
    use_rerank: bool = False,
    top_k: int = RERANK_TOP_K,
    verbose: bool = True,
    print_summary: bool = True,
    search_mode: str = "vector",
    alpha: float = HYBRID_ALPHA,
    hybrid_score_threshold: float = HYBRID_SCORE_THRESHOLD,
) -> dict:
    """
    Evaluates one configuration (threshold + reranking) across the provided eval_set.

    Hit detection requires:
      - Page in expected_pages
      - Source doc name matches expected_doc
      - At least one required_keyword found in chunk content

    Multi-hop/multi-page questions additionally track complete evidence coverage.
    """
    if use_rerank and not verify_cohere_functional():
        raise RuntimeError(
            "Cohere API is unavailable or COHERE_API_KEY is missing/invalid. "
            "Cannot report fallback results as successful reranking."
        )

    first_ranks: list[int | None] = []
    keyword_hits: list[bool] = []
    evidence_coverages: list[tuple[int, int]] = []
    latencies_ms: list[float] = []
    oos_total = 0
    oos_rejected = 0
    failed_questions: list[dict] = []

    for item in eval_set:
        qid = item["id"]
        q = item["question"]
        expected_pages = item.get("expected_pages", [])
        expected_doc = item.get("expected_doc")
        required_keywords = item.get("required_keywords", [])
        is_oos = item.get("out_of_scope", False)
        q_type = item.get("type", "unknown")

        t0 = time.perf_counter()
        pipeline_kwargs = dict(
            question=q,
            tenant_id=tenant_id,
            source_filter=None,
            use_rerank=use_rerank,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        if (search_mode or "vector").strip().lower() == "hybrid":
            pipeline_kwargs["search_mode"] = "hybrid"
            pipeline_kwargs["alpha"] = alpha
            pipeline_kwargs["hybrid_score_threshold"] = hybrid_score_threshold
        pipeline_result = prepare_rag_pipeline(**pipeline_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        chunks = pipeline_result["selected_chunks"]
        retrieved_pages = [c.get("page") for c in chunks]
        retrieved_docs = [c.get("source") for c in chunks]

        if is_oos:
            oos_total += 1
            correctly_rejected = not pipeline_result["has_chunks"]
            if correctly_rejected:
                oos_rejected += 1
            if verbose:
                status = "PASS" if correctly_rejected else "FAIL"
                action = (
                    'Triggered "I don\'t know"'
                    if correctly_rejected
                    else f"Retrieved {len(chunks)} chunks (Hallucination risk)"
                )
                print(f"[{status}] Q{qid:02d}[{q_type:12s}]: {q}")
                print(f"       OOS Expected: rejected | Got: {retrieved_pages} | {action}", flush=True)
        else:
            # --- Richer hit detection ---
            first_hit_rank = compute_hit_rank(
                chunks=chunks,
                expected_pages=expected_pages,
                expected_doc=expected_doc,
                required_keywords=required_keywords,
            )
            first_ranks.append(first_hit_rank)

            # Keyword-validated: at least one chunk has page+doc+keyword match
            kw_ok = first_hit_rank is not None
            keyword_hits.append(kw_ok)

            # Evidence coverage (for multi_hop and multi_page)
            is_multi = q_type in ("multi_hop", "multi_page")
            if is_multi:
                found, required = compute_evidence_coverage(chunks, expected_pages, expected_doc)
                evidence_coverages.append((found, required))
            else:
                evidence_coverages.append((0, 0))  # placeholder

            if verbose:
                rank_str = str(first_hit_rank) if first_hit_rank is not None else "-"
                correct_pages = sum(
                    1 for c in chunks
                    if c.get("page") in expected_pages and _doc_name_hit(c, expected_doc)
                )
                total_retrieved = len(chunks)
                prec = (correct_pages / total_retrieved * 100) if total_retrieved > 0 else 0.0
                status = "PASS" if first_hit_rank is not None else "FAIL"
                evidence_note = ""
                if is_multi:
                    found, required = evidence_coverages[-1]
                    evidence_note = f" | Evidence: {found}/{required} pages"
                kw_note = " | KW:✓" if kw_ok else " | KW:✗"
                print(f"[{status}] Q{qid:02d}[{q_type:12s}]: {q}")
                print(
                    f"       Expected doc={expected_doc} pages={expected_pages} "
                    f"| Got pages={retrieved_pages} | HitRank={rank_str} "
                    f"| Prec@{top_k}={prec:.0f}%{kw_note}{evidence_note} "
                    f"| {elapsed_ms:.0f}ms",
                    flush=True,
                )

            if first_hit_rank is None:
                failed_questions.append({
                    "id": qid,
                    "type": q_type,
                    "question": q,
                    "expected_pages": expected_pages,
                    "expected_doc": expected_doc,
                    "retrieved_pages": retrieved_pages,
                    "retrieved_docs": list(set(retrieved_docs)),
                    "required_keywords": required_keywords,
                })

        # Rate-limit safety: Cohere free tier 10 req/min
        if use_rerank and not is_oos:
            time.sleep(6.1)

    metrics = calculate_evaluation_metrics(
        first_ranks=first_ranks,
        oos_total=oos_total,
        oos_rejected=oos_rejected,
        keyword_hits=keyword_hits,
        evidence_coverages=evidence_coverages,
        latencies_ms=latencies_ms,
    )
    metrics["threshold"] = distance_threshold
    metrics["reranking"] = "ON" if use_rerank else "OFF"
    metrics["search_mode"] = search_mode
    metrics["alpha"] = alpha if search_mode == "hybrid" else None
    metrics["failed_questions"] = failed_questions

    if print_summary:
        print_evaluation_summary(metrics)

    return metrics


# ---------------------------------------------------------------------------
# 6-Configuration Comparison
# ---------------------------------------------------------------------------

def run_comparison(
    tenant_id: str,
    eval_set: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """Runs all 6 configurations: 3 thresholds × rerank OFF/ON."""
    n_in_scope = sum(1 for q in eval_set if not q.get("out_of_scope"))
    n_oos = sum(1 for q in eval_set if q.get("out_of_scope"))
    print(f"\n{'=' * 90}", flush=True)
    print(f" RAG HARDER EVALUATION — 6 CONFIGURATIONS", flush=True)
    print(f" Tenant: {tenant_id}", flush=True)
    print(f" Dataset: {len(eval_set)} questions ({n_in_scope} in-scope, {n_oos} out-of-scope)", flush=True)
    print(f" Stage-1 Retrieval: {RETRIEVAL_TOP_K} chunks | Final Top-K: {top_k}", flush=True)
    print(f" Thresholds: {COMPARISON_THRESHOLDS} | Reranker: Cohere rerank-english-v3.0", flush=True)
    print(f" Hit detection: doc + page + keyword match", flush=True)
    print(f"{'=' * 90}\n", flush=True)

    results = []
    for use_rerank in [False, True]:
        for threshold in COMPARISON_THRESHOLDS:
            state = "ON" if use_rerank else "OFF"
            print(
                f"\n--- Config: threshold={threshold:.2f} | Cohere={state} ---",
                flush=True,
            )
            res = evaluate_single_configuration(
                tenant_id=tenant_id,
                eval_set=eval_set,
                distance_threshold=threshold,
                use_rerank=use_rerank,
                top_k=top_k,
                verbose=True,
                print_summary=False,
            )
            results.append(res)
            kw_note = (
                f" | KW@4={res['keyword_validated_hit_at_4']:.2f}%"
                if res["keyword_validated_hit_at_4"] is not None
                else ""
            )
            ev_note = (
                f" | Evidence={res['complete_evidence_rate']:.2f}%"
                if res["complete_evidence_rate"] is not None
                else ""
            )
            lat_note = (
                f" | Avg={res['avg_latency_ms']:.0f}ms"
                if res["avg_latency_ms"] is not None
                else ""
            )
            print(
                f"  → Hit@1={res['hit_at_1']:.2f}% | Hit@4={res['hit_at_4']:.2f}% "
                f"| MRR={res['mrr']:.4f} | OOS={res['rejection_rate']:.2f}%"
                f"{kw_note}{ev_note}{lat_note}\n",
                flush=True,
            )

    print_comparison_table(results)
    return results


# ---------------------------------------------------------------------------
# Comparison Table Printer
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]):
    """Prints the final comparison table in Markdown format and derives recommendation."""
    print("\n" + "=" * 100)
    print(" RAG HARDER EVALUATION — COMPARISON TABLE (6 CONFIGURATIONS)")
    print("=" * 100 + "\n")

    header = (
        "| Threshold | Reranking | Hit@1 | Hit@4 | MRR   | OOS Reject | KW@4  | Evidence | Avg Latency |"
    )
    divider = "|-----------|-----------|-------|-------|-------|------------|-------|----------|-------------|"
    print(header)
    print(divider)

    for r in results:
        kw = f"{r['keyword_validated_hit_at_4']:.1f}%" if r["keyword_validated_hit_at_4"] is not None else "N/A"
        ev = f"{r['complete_evidence_rate']:.1f}%" if r["complete_evidence_rate"] is not None else "N/A"
        lat = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] is not None else "N/A"
        print(
            f"| {r['threshold']:.2f}      | {r['reranking']:9s} "
            f"| {r['hit_at_1']:5.1f}% | {r['hit_at_4']:5.1f}% "
            f"| {r['mrr']:.3f} | {r['rejection_rate']:9.1f}% "
            f"| {kw:5s} | {ev:8s} | {lat:11s} |"
        )

    # --- Data-driven recommendation ---
    # Score = rejection_rate weight (heavy) + mrr + hit_at_4/100 + rerank_bonus
    def config_score(r: dict) -> float:
        oos = r["rejection_rate"]
        # Penalise if OOS rejection is not 100%
        oos_penalty = 0.0 if oos >= 100.0 else (100.0 - oos) * 2
        rerank_bonus = 0.5 if r["reranking"] == "ON" else 0.0
        ev_bonus = (r["complete_evidence_rate"] or 0.0) / 100.0
        return r["mrr"] + r["hit_at_4"] / 100.0 + rerank_bonus + ev_bonus - oos_penalty / 100.0

    best = max(results, key=config_score)

    print("\n" + "=" * 100)
    print(f"Recommended Configuration: threshold={best['threshold']:.2f} | Cohere Reranking={best['reranking']}")
    print(
        f"Performance: Hit@1={best['hit_at_1']:.2f}% | Hit@4={best['hit_at_4']:.2f}% "
        f"| MRR={best['mrr']:.4f} | OOS Rejection={best['rejection_rate']:.2f}%"
    )
    # --- Data-driven justification ---
    _print_recommendation_rationale(results, best)
    print("=" * 100 + "\n")

    # --- Failed question analysis ---
    _print_failed_analysis(results)


def _print_recommendation_rationale(results: list[dict], best: dict):
    """Derives a data-driven recommendation from actual benchmark results."""
    all_oos_100 = [r for r in results if r["rejection_rate"] >= 100.0]
    rerank_on = [r for r in results if r["reranking"] == "ON"]
    rerank_off = [r for r in results if r["reranking"] == "OFF"]

    avg_hit4_on = sum(r["hit_at_4"] for r in rerank_on) / len(rerank_on) if rerank_on else 0.0
    avg_hit4_off = sum(r["hit_at_4"] for r in rerank_off) / len(rerank_off) if rerank_off else 0.0
    rerank_improvement = avg_hit4_on - avg_hit4_off

    thresholds_with_100_oos = sorted({r["threshold"] for r in all_oos_100})

    lines = []

    if thresholds_with_100_oos:
        lines.append(
            f"Thresholds achieving 100% OOS rejection: {thresholds_with_100_oos}. "
            f"Selected threshold={best['threshold']:.2f} as it maximises retrieval quality among these."
        )
    else:
        lines.append(
            "No threshold achieved 100% OOS rejection in this run. "
            f"Threshold={best['threshold']:.2f} offers the best balance of retrieval and rejection."
        )

    if abs(rerank_improvement) < 1.0:
        lines.append(
            f"Cohere reranking shows <1% improvement in Hit@4 ({rerank_improvement:+.1f}pp) "
            "— marginal benefit; not recommended if Cohere quota is limited."
        )
    elif rerank_improvement > 0:
        lines.append(
            f"Cohere reranking improves Hit@4 by {rerank_improvement:.1f}pp on average — "
            "measurable benefit; recommended if latency budget allows."
        )
    else:
        lines.append(
            f"Cohere reranking reduced Hit@4 by {-rerank_improvement:.1f}pp — "
            "vector search ordering outperformed cross-encoder on this dataset."
        )

    for line in lines:
        print(f"Rationale: {line}")


def _print_failed_analysis(results: list[dict]):
    """Prints failed question analysis across all 6 configurations."""
    # Collect questions that failed in ALL configurations
    all_failed_ids: dict[int, dict] = {}
    id_fail_count: dict[int, int] = {}

    for r in results:
        for fq in r.get("failed_questions", []):
            qid = fq["id"]
            id_fail_count[qid] = id_fail_count.get(qid, 0) + 1
            all_failed_ids[qid] = fq

    if not all_failed_ids:
        print("\n[+] No questions failed across all 6 configurations — excellent retrieval coverage!\n")
        return

    print("\n--- Failed Question Analysis ---")
    print(f"{'ID':>4} | {'Type':12s} | {'Fail/6':6s} | {'Expected Pages':20s} | Question")
    print("-" * 100)
    for qid in sorted(all_failed_ids):
        fq = all_failed_ids[qid]
        fc = id_fail_count[qid]
        pages = str(fq["expected_pages"])
        q_short = fq["question"][:60] + "…" if len(fq["question"]) > 60 else fq["question"]
        print(f"{qid:>4} | {fq['type']:12s} | {fc:>6}/6 | {pages:20s} | {q_short}")
    print()


# ---------------------------------------------------------------------------
# Single-config entry point (used by --threshold / --rerank modes)
# ---------------------------------------------------------------------------

def run_evaluation(
    tenant_id: str,
    eval_set: list[dict],
    top_k: int = RERANK_TOP_K,
    use_rerank: bool = False,
    distance_threshold: float = DISTANCE_THRESHOLD,
    search_mode: str = SEARCH_MODE,
    alpha: float = HYBRID_ALPHA,
) -> dict:
    """Evaluates one configuration and prints verbose results."""
    return evaluate_single_configuration(
        tenant_id=tenant_id,
        eval_set=eval_set,
        distance_threshold=distance_threshold,
        use_rerank=use_rerank,
        top_k=top_k,
        verbose=True,
        print_summary=True,
        search_mode=search_mode,
        alpha=alpha,
    )


def run_hybrid_comparison(
    tenant_id: str,
    eval_set: list[dict],
    top_k: int = RERANK_TOP_K,
    distance_threshold: float = 0.40,
) -> list[dict]:
    """Compares vector-only baselines (0.30 & 0.40) against hybrid search (alpha=0.25, 0.50, 0.75) with rerank OFF/ON."""
    configs = [
        ("vector", None, 0.30, False),
        ("vector", None, 0.30, True),
        ("vector", None, 0.40, False),
        ("vector", None, 0.40, True),
    ]
    for a in HYBRID_ALPHAS:
        configs.append(("hybrid", a, distance_threshold, False))
        configs.append(("hybrid", a, distance_threshold, True))

    print(f"\n{'=' * 105}", flush=True)
    print(" VECTOR-ONLY VS HYBRID SEARCH (BM25 + VECTOR) COMPARISON", flush=True)
    print(f" Tenant: {tenant_id} | Dataset: {len(eval_set)} Qs | Stage-1: {RETRIEVAL_TOP_K} | Top-K: {top_k}", flush=True)
    print(f" Hybrid Alphas: {HYBRID_ALPHAS} | Max Vector Distance: {distance_threshold:.2f}", flush=True)
    print(f"{'=' * 105}\n", flush=True)

    results = []
    for mode, alpha_val, thresh, use_rerank in configs:
        label = f"mode={mode}" + (f" alpha={alpha_val:.2f}" if alpha_val is not None else "")
        print(f"\n--- Config: {label} | max_dist={thresh:.2f} | Cohere={'ON' if use_rerank else 'OFF'} ---", flush=True)
        res = evaluate_single_configuration(
            tenant_id=tenant_id,
            eval_set=eval_set,
            distance_threshold=thresh,
            use_rerank=use_rerank,
            top_k=top_k,
            verbose=True,
            print_summary=False,
            search_mode=mode,
            alpha=alpha_val if alpha_val is not None else HYBRID_ALPHA,
        )
        results.append(res)

    print("\n" + "=" * 112)
    print(" VECTOR-ONLY VS HYBRID SEARCH — COMPARISON TABLE")
    print("=" * 112)
    print("| Mode   | Alpha | MaxDist | Rerank | Hit@1 | Hit@4 | MRR   | OOS Reject | Evidence | Avg Latency | Q15 |")
    print("|--------|-------|---------|--------|-------|-------|-------|------------|----------|-------------|-----|")
    for r in results:
        alpha_str = f"{r['alpha']:.2f}" if r.get("alpha") is not None else "N/A  "
        ev = f"{r['complete_evidence_rate']:.1f}%" if r["complete_evidence_rate"] is not None else "N/A"
        lat = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] is not None else "N/A"
        q15_failed = any(fq["id"] == 15 for fq in r.get("failed_questions", []))
        q15_str = "✗" if q15_failed else "✓"
        print(
            f"| {r['search_mode']:6s} | {alpha_str:5s} | {r['threshold']:.2f}    | {r['reranking']:6s} "
            f"| {r['hit_at_1']:5.1f}% | {r['hit_at_4']:5.1f}% | {r['mrr']:.3f} "
            f"| {r['rejection_rate']:9.1f}% | {ev:8s} | {lat:11s} | {q15_str:3s} |"
        )
    print("=" * 112 + "\n")
    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Harder RAG Evaluation — doc+page+keyword hit detection, 6-config & hybrid comparison"
    )
    parser.add_argument("--tenant", type=str, default=None, help="Weaviate tenant ID")
    parser.add_argument(
        "--top-k", type=int, default=RERANK_TOP_K,
        help=f"Final top-k chunks (default: RERANK_TOP_K={RERANK_TOP_K})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DISTANCE_THRESHOLD,
        help=f"Distance threshold (default: DISTANCE_THRESHOLD={DISTANCE_THRESHOLD})"
    )
    parser.add_argument("--rerank", action="store_true", help="Enable Cohere reranking")
    parser.add_argument(
        "--search-mode", type=str, default=SEARCH_MODE, choices=["vector", "hybrid"],
        help=f"Retrieval mode: vector or hybrid (default: {SEARCH_MODE})"
    )
    parser.add_argument(
        "--alpha", type=float, default=HYBRID_ALPHA,
        help=f"Hybrid alpha weight (0=BM25, 1=Vector, default: {HYBRID_ALPHA})"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run all 6 configurations (3 thresholds × rerank OFF/ON)"
    )
    parser.add_argument(
        "--compare-hybrid", action="store_true",
        help="Compare vector-only (0.30, 0.40) vs hybrid (alpha 0.25, 0.50, 0.75) with rerank OFF/ON"
    )
    parser.add_argument("--ingest", action="store_true", help="Force re-ingest PDF before evaluation")
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
        help=f"Path to evaluation dataset JSON (default: {DEFAULT_DATASET_PATH})"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    eval_set = load_eval_dataset(dataset_path)
    print(f"[+] Loaded {len(eval_set)} evaluation questions from '{dataset_path.name}'", flush=True)

    tenant = resolve_tenant(args.tenant)

    if args.ingest:
        pages = load_pdf(PDF_PATH)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        vectors = embed_documents([c["content"] for c in chunks])
        ingest_chunks(chunks=chunks, source=PDF_PATH.name, vectors=vectors, tenant_id=tenant)
    else:
        ensure_document_ingested(tenant_id=tenant, pdf_file=PDF_PATH)

    try:
        if args.compare_hybrid:
            run_hybrid_comparison(
                tenant_id=tenant,
                eval_set=eval_set,
                top_k=args.top_k,
                distance_threshold=args.threshold if args.threshold != DISTANCE_THRESHOLD else 0.40,
            )
        elif args.compare:
            run_comparison(tenant_id=tenant, eval_set=eval_set, top_k=args.top_k)
        else:
            print(f"\n{'=' * 80}", flush=True)
            print(f" RAG HARDER RETRIEVAL EVALUATION", flush=True)
            print(
                f" Tenant: {tenant} | Dataset: {len(eval_set)} Qs | Mode: {args.search_mode} "
                f"| Alpha: {args.alpha:.2f} | Top-K: {args.top_k} "
                f"| Reranker: {'ON' if args.rerank else 'OFF'} | Threshold: {args.threshold:.2f}",
                flush=True,
            )
            print(f"{'=' * 80}\n", flush=True)
            run_evaluation(
                tenant_id=tenant,
                eval_set=eval_set,
                top_k=args.top_k,
                use_rerank=args.rerank,
                distance_threshold=args.threshold,
                search_mode=args.search_mode,
                alpha=args.alpha,
            )
    finally:
        close_weaviate_client()


if __name__ == "__main__":
    main()
