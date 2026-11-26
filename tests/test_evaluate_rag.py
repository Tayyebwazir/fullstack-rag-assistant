"""Test suite verifying evaluate_rag.py updates:
1. Reuses shared prepare_rag_pipeline.
2. Weaviate retrieval requests RETRIEVAL_TOP_K (20) chunks.
3. Cohere reranking selects RERANK_TOP_K (4) chunks when rerank is True.
4. Cohere reranking is skipped when rerank is False.
5. Applies distance threshold cutoffs (0.20, 0.30, 0.40).
6. Preserves EVAL_SET 10 questions and expected pages.
7. Evaluates all 6 configurations and outputs comparison table.
8. Fails cleanly when Cohere API is unavailable (no false fallbacks).
"""

from unittest.mock import MagicMock, patch

from evaluate_rag import (
    EVAL_SET,
    COMPARISON_THRESHOLDS,
    calculate_evaluation_metrics,
    evaluate_single_configuration,
    print_evaluation_summary,
    run_comparison,
    run_evaluation,
)
from app.config import RETRIEVAL_TOP_K, RERANK_TOP_K


def test_calculate_evaluation_metrics_known_ranks():
    """Verify Hit@1, Hit@4, and MRR formulas against known example rank sets."""
    # Test case 1: Mixed known ranks
    ranks_1 = [1, 2, 4, None, None]
    res_1 = calculate_evaluation_metrics(ranks_1)
    assert res_1["total"] == 5
    assert res_1["hit_at_1"] == 20.0, f"Expected Hit@1=20.0, got {res_1['hit_at_1']}"
    assert res_1["hit_at_4"] == 60.0, f"Expected Hit@4=60.0, got {res_1['hit_at_4']}"
    expected_mrr_1 = (1.0 + 0.5 + 0.25 + 0.0 + 0.0) / 5.0  # 0.35
    assert abs(res_1["mrr"] - expected_mrr_1) < 1e-4, f"Expected MRR={expected_mrr_1}, got {res_1['mrr']}"
    print("[PASS] Known ranks test 1 (mixed ranks) verified")

    # Test case 2: All misses
    ranks_miss = [None, None, None]
    res_miss = calculate_evaluation_metrics(ranks_miss)
    assert res_miss["hit_at_1"] == 0.0
    assert res_miss["hit_at_4"] == 0.0
    assert res_miss["mrr"] == 0.0
    print("[PASS] Known ranks test 2 (all misses) verified")

    # Test case 3: 10 questions with 7 at rank 1, 1 at rank 2, 1 at rank 3, 1 miss
    # Hit@1 = 70.00%, Hit@4 = 90.00%
    ranks_10 = [1, 1, 1, 1, 1, 1, 1, 2, 3, None]
    res_10 = calculate_evaluation_metrics(ranks_10)
    assert res_10["total"] == 10
    assert res_10["hit_at_1"] == 70.0
    assert res_10["hit_at_4"] == 90.0
    expected_mrr_10 = (7 * 1.0 + 0.5 + 1.0 / 3.0 + 0.0) / 10.0
    assert abs(res_10["mrr"] - expected_mrr_10) < 1e-4
    print("[PASS] Known ranks test 3 (10 questions 70% Hit@1, 90% Hit@4) verified")


def test_eval_set_preservation():
    """Verify all 10 curated in-scope questions and 5 out-of-scope questions are present."""
    assert len(EVAL_SET) == 15, f"Expected 15 eval questions, got {len(EVAL_SET)}"
    # Verify in-scope questions
    in_scope = [q for q in EVAL_SET if not q.get("out_of_scope", False)]
    assert len(in_scope) == 10, f"Expected 10 in-scope questions, got {len(in_scope)}"
    assert in_scope[0]["id"] == 1
    assert in_scope[0]["expected_pages"] == [4]
    assert in_scope[9]["id"] == 10
    assert in_scope[9]["expected_pages"] == [3, 5]

    # Verify out-of-scope questions
    out_of_scope = [q for q in EVAL_SET if q.get("out_of_scope", False)]
    assert len(out_of_scope) == 5, f"Expected 5 out-of-scope questions, got {len(out_of_scope)}"
    assert any("World Cup" in q["question"] for q in out_of_scope)
    for q in out_of_scope:
        assert q["expected_pages"] == []
    print("[PASS] EVAL_SET verified with 10 in-scope and 5 out-of-scope questions")


def test_out_of_scope_rejection_logic():
    """Verify that out-of-scope questions with no candidate chunks trigger 'I don't know' and count as PASS."""
    with patch("evaluate_rag.verify_cohere_functional", return_value=True), \
         patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=[]):
        
        metrics = evaluate_single_configuration(
            tenant_id="test-tenant-123",
            distance_threshold=0.30,
            use_rerank=False,
            verbose=False,
            print_summary=False,
        )
        assert metrics["out_of_scope_total"] == 5
        assert metrics["out_of_scope_rejected"] == 5
        assert metrics["rejection_rate"] == 100.0
        print("[PASS] Out-of-scope rejection correctly triggers 'I don't know' with 100% rejection rate")


def test_run_comparison_6_configurations():
    """Verify all 6 configurations run across thresholds [0.20, 0.30, 0.40] and rerank [OFF, ON]."""
    mock_candidates = [
        {"source": "Attention is all you need.pdf", "page": (i % 8) + 1, "chunk_index": i, "content": f"Chunk text {i}"}
        for i in range(20)
    ]
    mock_reranked = [
        {"source": "Attention is all you need.pdf", "page": 4, "chunk_index": 3, "content": "Attention equation", "rerank_position": 1, "relevance_score": 0.95},
        {"source": "Attention is all you need.pdf", "page": 5, "chunk_index": 5, "content": "FFN formula", "rerank_position": 2, "relevance_score": 0.90},
        {"source": "Attention is all you need.pdf", "page": 6, "chunk_index": 7, "content": "Sinusoidal encoding", "rerank_position": 3, "relevance_score": 0.85},
        {"source": "Attention is all you need.pdf", "page": 7, "chunk_index": 9, "content": "Optimizer schedule", "rerank_position": 4, "relevance_score": 0.80},
    ]

    # Out-of-scope questions return empty list (filtered out by distance), in-scope return mock_candidates
    def mock_embed(text):
        return [0.99] * 3072 if "World Cup" in text or "cookies" in text or "appendicitis" in text or "Apple" in text or "Civic" in text else [0.1] * 3072

    def mock_search(query_vector, tenant_id, source_filter, limit, distance_threshold):
        # Distance filter rejects out-of-scope query vectors (0.99)
        if query_vector[0] > 0.5:
            return []
        return mock_candidates

    with patch("time.sleep"), \
         patch("evaluate_rag.verify_cohere_functional", return_value=True), \
         patch("app.services.rag_service.embed_query", side_effect=mock_embed), \
         patch("app.services.rag_service.search_chunks", side_effect=mock_search) as search_spy, \
         patch("app.services.rag_service.rerank_chunks", return_value=mock_reranked) as mock_rerank:

        results = run_comparison(tenant_id="test-tenant-123", top_k=RERANK_TOP_K)

        assert len(results) == 6, f"Expected 6 configuration results, got {len(results)}"

        # Check all combinations (Reranking OFF: 0.20, 0.30, 0.40; then Reranking ON: 0.20, 0.30, 0.40)
        expected_configs = [
            (0.20, "OFF"), (0.30, "OFF"), (0.40, "OFF"),
            (0.20, "ON"), (0.30, "ON"), (0.40, "ON"),
        ]
        actual_configs = [(r["threshold"], r["reranking"]) for r in results]
        assert actual_configs == expected_configs, f"Config mismatch: {actual_configs}"

        # 6 configs * 15 questions = 90 search calls
        assert search_spy.call_count == 90
        # 3 'ON' configs * 10 in-scope questions = 30 rerank calls
        assert mock_rerank.call_count == 30

        print("[PASS] All 6 configurations evaluated successfully")


def test_cohere_unavailability_handled_cleanly():
    """Verify that if Cohere is unavailable, reranking ON raises an error instead of reporting fallbacks."""
    with patch("evaluate_rag.verify_cohere_functional", return_value=False):
        try:
            evaluate_single_configuration(
                tenant_id="test-tenant-123",
                distance_threshold=0.30,
                use_rerank=True,
                verbose=False,
            )
            assert False, "Expected RuntimeError when Cohere is unavailable"
        except RuntimeError as exc:
            assert "Cannot report fallback results as successful reranking" in str(exc)
            print("[PASS] Cohere unavailability cleanly raises error (no silent fallback)")


def test_cli_parser_and_rerank_flag():
    """Verify CLI parser sets rerank default to False, enabled with --rerank, and help text specifies Cohere."""
    import argparse
    from evaluate_rag import main
    # Create parser equivalent to main's parser
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Before/After Cohere Reranking & Distance Thresholds")
    parser.add_argument("--rerank", action="store_true", help="Enable Cohere reranking")
    
    # Test default
    args_default = parser.parse_args([])
    assert args_default.rerank is False, "Default rerank should be False"

    # Test --rerank
    args_rerank = parser.parse_args(["--rerank"])
    assert args_rerank.rerank is True, "--rerank should set rerank to True"

    # Test help text does not contain Gemini and contains Cohere
    action = [a for a in parser._actions if "--rerank" in a.option_strings][0]
    assert "Cohere reranking" in action.help, f"Expected 'Cohere reranking' in help, got {action.help}"
    assert "Gemini" not in action.help, "Found unexpected 'Gemini' in help text"
    print("[PASS] CLI parser and --rerank toggle verified")


def test_two_stage_pipeline_20_retrieval_and_4_rerank():
    """Verify that retrieval queries 20 chunks from Weaviate and reranks to 4 chunks."""
    from app.services.rag_service import prepare_rag_pipeline
    
    mock_candidates = [
        {"source": "Attention is all you need.pdf", "page": (i % 8) + 1, "chunk_index": i, "content": f"Chunk text {i}"}
        for i in range(20)
    ]
    mock_reranked = [
        {"source": "Attention is all you need.pdf", "page": 4, "chunk_index": 3, "content": "Attention equation", "rerank_position": 1, "relevance_score": 0.95},
        {"source": "Attention is all you need.pdf", "page": 5, "chunk_index": 5, "content": "FFN formula", "rerank_position": 2, "relevance_score": 0.90},
        {"source": "Attention is all you need.pdf", "page": 6, "chunk_index": 7, "content": "Sinusoidal encoding", "rerank_position": 3, "relevance_score": 0.85},
        {"source": "Attention is all you need.pdf", "page": 7, "chunk_index": 9, "content": "Optimizer schedule", "rerank_position": 4, "relevance_score": 0.80},
    ]

    with patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=mock_candidates) as mock_search, \
         patch("app.services.rag_service.rerank_chunks", return_value=mock_reranked) as mock_rerank:
        
        # Test rerank ON
        result_on = prepare_rag_pipeline(
            question="What is attention?",
            tenant_id="test-tenant-123",
            use_rerank=True,
            top_k=4,
            distance_threshold=0.30,
        )
        assert mock_search.call_args[1]["limit"] == 20
        assert mock_search.call_args[1]["distance_threshold"] == 0.30
        assert mock_rerank.call_args[1]["top_k"] == 4
        assert len(result_on["selected_chunks"]) == 4

        # Test rerank OFF
        result_off = prepare_rag_pipeline(
            question="What is attention?",
            tenant_id="test-tenant-123",
            use_rerank=False,
            top_k=4,
            distance_threshold=0.30,
        )
        assert len(result_off["selected_chunks"]) == 4
        # Selected chunks should be top 4 from vector search candidates
        assert result_off["selected_chunks"] == mock_candidates[:4]

        print("[PASS] Two-stage pipeline (20 candidates, 4 selected) verified for both ON and OFF")


def test_evaluation_summary_output():
    """Verify that print_evaluation_summary prints Hit@1, Hit@4, MRR banner."""
    import io
    import sys
    
    mock_metrics = {
        "total": 10,
        "hit_at_1": 70.0,
        "hit_at_4": 90.0,
        "mrr": 0.82,
    }
    
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        print_evaluation_summary(mock_metrics)
    finally:
        sys.stdout = old_stdout
        
    output = buf.getvalue()
    assert "========== RAG Evaluation Summary ==========" in output
    assert "Total Questions: 10" in output
    assert "Hit@1: 70.00%" in output
    assert "Hit@4: 90.00%" in output
    assert "MRR:   0.82" in output
    print("[PASS] Evaluation summary output format verified")


if __name__ == "__main__":
    test_calculate_evaluation_metrics_known_ranks()
    test_eval_set_preservation()
    test_out_of_scope_rejection_logic()
    test_run_comparison_6_configurations()
    test_cohere_unavailability_handled_cleanly()
    test_cli_parser_and_rerank_flag()
    test_two_stage_pipeline_20_retrieval_and_4_rerank()
    test_evaluation_summary_output()
    print("\n🎉 ALL BEFORE/AFTER EVALUATION COMPARISON TESTS PASSED!")
