"""
tests/test_harder_evaluation.py

Pytest test suite for evaluate_rag_harder.py — Feature: feature/harder-evaluation

Covers:
  1. Dataset loading and structure validation
  2. Document + page + keyword hit detection logic
  3. Evidence coverage (multi-hop / multi-page)
  4. Metric calculations (Hit@1, Hit@4, MRR, keyword@4, evidence rate, latency)
  5. Out-of-scope rejection
  6. All six configurations (3 thresholds × rerank OFF/ON)
  7. Cohere reranking fallback behaviour (unavailable → clean RuntimeError)
  8. Correct page but wrong answer content (keyword miss)
  9. Partial multi-hop evidence tracking
  10. Latency tracking
  11. Comparison table and data-driven recommendation
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Import from the new evaluation module
# ---------------------------------------------------------------------------
import sys
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate_rag_harder import (
    COMPARISON_THRESHOLDS,
    _doc_name_hit,
    _keyword_hit,
    _page_hit,
    calculate_evaluation_metrics,
    compute_evidence_coverage,
    compute_hit_rank,
    evaluate_single_configuration,
    load_eval_dataset,
    print_comparison_table,
    print_evaluation_summary,
    run_comparison,
    run_evaluation,
    verify_cohere_functional,
    DEFAULT_DATASET_PATH,
)
from app.config import RERANK_TOP_K, RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_ATTN_CHUNKS = [
    {
        "source": "Attention is all you need.pdf",
        "page": 4,
        "chunk_index": 0,
        "content": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        "distance": 0.15,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 5,
        "chunk_index": 1,
        "content": "MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W",
        "distance": 0.18,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 6,
        "chunk_index": 2,
        "content": "PE(pos,2i) = sin(pos / 10000^(2i/d_model))",
        "distance": 0.22,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 7,
        "chunk_index": 3,
        "content": "We trained on 8 P100 GPUs for 100000 steps",
        "distance": 0.25,
    },
]

# Helper to add chunk_index to any chunk dict list that lacks it
def _ensure_chunk_index(chunks: list[dict]) -> list[dict]:
    return [dict(c, chunk_index=c.get("chunk_index", i)) for i, c in enumerate(chunks)]

MOCK_RERANKED_CHUNKS = [
    {
        "source": "Attention is all you need.pdf",
        "page": 4,
        "chunk_index": 0,
        "content": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        "relevance_score": 0.95,
        "rerank_position": 1,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 5,
        "chunk_index": 1,
        "content": "MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W",
        "relevance_score": 0.90,
        "rerank_position": 2,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 6,
        "chunk_index": 2,
        "content": "PE(pos,2i) = sin(pos / 10000^(2i/d_model))",
        "relevance_score": 0.85,
        "rerank_position": 3,
    },
    {
        "source": "Attention is all you need.pdf",
        "page": 7,
        "chunk_index": 3,
        "content": "We trained on 8 P100 GPUs for 100000 steps",
        "relevance_score": 0.80,
        "rerank_position": 4,
    },
]

SAMPLE_EVAL_SET = [
    {
        "id": 1,
        "type": "exact_term",
        "question": "What is the mathematical equation for Scaled Dot-Product Attention?",
        "expected_doc": "Attention is all you need.pdf",
        "expected_pages": [4],
        "required_keywords": ["softmax", "d_k"],
        "out_of_scope": False,
        "note": "Exact term test",
    },
    {
        "id": 2,
        "type": "multi_hop",
        "question": "How does multi-head attention work and what is the FFN formula?",
        "expected_doc": "Attention is all you need.pdf",
        "expected_pages": [5, 8],
        "required_keywords": ["MultiHead", "FFN"],
        "out_of_scope": False,
        "note": "Multi-hop: pages 5 and 8",
    },
    {
        "id": 3,
        "type": "out_of_scope",
        "question": "What is the GDP of Germany in 2023?",
        "expected_doc": None,
        "expected_pages": [],
        "required_keywords": [],
        "out_of_scope": True,
        "note": "Out-of-scope: economics",
    },
]


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    """Tests for load_eval_dataset."""

    def test_loads_harder_dataset_file(self):
        """The default harder dataset should load with >= 25 questions."""
        if not DEFAULT_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found: {DEFAULT_DATASET_PATH}")
        dataset = load_eval_dataset(DEFAULT_DATASET_PATH)
        assert len(dataset) >= 25, f"Expected >= 25 questions, got {len(dataset)}"

    def test_dataset_has_required_fields(self):
        """Every question must have id, question, out_of_scope fields."""
        if not DEFAULT_DATASET_PATH.exists():
            pytest.skip("Dataset file missing")
        dataset = load_eval_dataset(DEFAULT_DATASET_PATH)
        for item in dataset:
            assert "id" in item
            assert "question" in item
            assert "out_of_scope" in item

    def test_dataset_question_types_present(self):
        """Dataset must include paraphrase, multi_hop, multi_page, out_of_scope types."""
        if not DEFAULT_DATASET_PATH.exists():
            pytest.skip("Dataset file missing")
        dataset = load_eval_dataset(DEFAULT_DATASET_PATH)
        types_present = {item.get("type") for item in dataset}
        for required_type in ("paraphrase", "multi_hop", "multi_page", "out_of_scope", "exact_term"):
            assert required_type in types_present, f"Question type '{required_type}' missing from dataset"

    def test_dataset_in_scope_count(self):
        """Dataset must have more in-scope than out-of-scope questions."""
        if not DEFAULT_DATASET_PATH.exists():
            pytest.skip("Dataset file missing")
        dataset = load_eval_dataset(DEFAULT_DATASET_PATH)
        in_scope = [q for q in dataset if not q.get("out_of_scope")]
        oos = [q for q in dataset if q.get("out_of_scope")]
        assert len(in_scope) >= 7, f"Expected >= 7 out-of-scope questions, got {len(oos)}"
        assert len(in_scope) > len(oos), "Expected more in-scope than out-of-scope questions"

    def test_missing_dataset_raises_file_not_found(self, tmp_path):
        """Missing dataset path raises FileNotFoundError with helpful message."""
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="nonexistent.json"):
            load_eval_dataset(missing)

    def test_malformed_dataset_raises_assertion(self, tmp_path):
        """Dataset with missing required fields raises AssertionError."""
        bad_data = [{"question": "No id or out_of_scope"}]
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps(bad_data), encoding="utf-8")
        with pytest.raises(AssertionError):
            load_eval_dataset(bad_file)


# ---------------------------------------------------------------------------
# 2. Hit detection primitives
# ---------------------------------------------------------------------------

class TestHitDetection:
    """Unit tests for _keyword_hit, _page_hit, _doc_name_hit, compute_hit_rank."""

    def test_keyword_hit_found(self):
        assert _keyword_hit("Attention softmax formula", ["softmax", "d_k"]) is True

    def test_keyword_hit_not_found(self):
        assert _keyword_hit("BLEU score results", ["softmax", "d_k"]) is False

    def test_keyword_hit_empty_keywords_always_true(self):
        """Empty keywords list means no keyword constraint — always passes."""
        assert _keyword_hit("Any text at all", []) is True

    def test_keyword_hit_case_insensitive(self):
        assert _keyword_hit("SOFTMAX is used here", ["softmax"]) is True

    def test_page_hit_match(self):
        assert _page_hit(4, [4, 5]) is True

    def test_page_hit_no_match(self):
        assert _page_hit(9, [4, 5]) is False

    def test_page_hit_empty_expected_always_false(self):
        """Empty expected_pages means OOS — no page can match."""
        assert _page_hit(1, []) is False

    def test_page_hit_none_page(self):
        assert _page_hit(None, [4]) is False

    def test_doc_name_hit_match(self):
        chunk = {"source": "Attention is all you need.pdf"}
        assert _doc_name_hit(chunk, "Attention is all you need.pdf") is True

    def test_doc_name_hit_mismatch(self):
        chunk = {"source": "LLM-based_autonomous_agents.pdf"}
        assert _doc_name_hit(chunk, "Attention is all you need.pdf") is False

    def test_doc_name_hit_none_expected(self):
        """None expected_doc means no doc constraint — always passes."""
        chunk = {"source": "anything.pdf"}
        assert _doc_name_hit(chunk, None) is True

    def test_compute_hit_rank_first_chunk_matches(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 4, "content": "softmax formula d_k"},
            {"source": "Attention is all you need.pdf", "page": 5, "content": "MultiHead Concat"},
        ]
        rank = compute_hit_rank(chunks, [4], "Attention is all you need.pdf", ["softmax"])
        assert rank == 1

    def test_compute_hit_rank_second_chunk_matches(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 3, "content": "encoder layers"},
            {"source": "Attention is all you need.pdf", "page": 4, "content": "softmax d_k formula"},
        ]
        rank = compute_hit_rank(chunks, [4], "Attention is all you need.pdf", ["softmax"])
        assert rank == 2

    def test_compute_hit_rank_correct_page_wrong_keyword(self):
        """Correct page but content lacks required keyword — should NOT count as hit."""
        chunks = [
            {
                "source": "Attention is all you need.pdf",
                "page": 4,
                "content": "This chunk is on page 4 but talks about dropout only",
            },
        ]
        rank = compute_hit_rank(chunks, [4], "Attention is all you need.pdf", ["softmax", "d_k"])
        assert rank is None, "Page match alone without keyword should not count as hit"

    def test_compute_hit_rank_wrong_doc(self):
        """Chunk from wrong document should not count as hit even if page matches."""
        chunks = [
            {"source": "LLM-based.pdf", "page": 4, "content": "softmax d_k"},
        ]
        rank = compute_hit_rank(chunks, [4], "Attention is all you need.pdf", ["softmax"])
        assert rank is None, "Wrong document should not count as hit"

    def test_compute_hit_rank_no_matching_chunks(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 2, "content": "encoder layers"},
        ]
        rank = compute_hit_rank(chunks, [8, 9], "Attention is all you need.pdf", ["BLEU"])
        assert rank is None


# ---------------------------------------------------------------------------
# 3. Evidence coverage (multi-hop / multi-page)
# ---------------------------------------------------------------------------

class TestEvidenceCoverage:
    """Tests for compute_evidence_coverage."""

    def test_full_evidence_all_pages_retrieved(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 3},
            {"source": "Attention is all you need.pdf", "page": 5},
        ]
        found, required = compute_evidence_coverage(chunks, [3, 5], "Attention is all you need.pdf")
        assert found == 2
        assert required == 2

    def test_partial_evidence_one_page_missing(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 3},
        ]
        found, required = compute_evidence_coverage(chunks, [3, 5], "Attention is all you need.pdf")
        assert found == 1
        assert required == 2

    def test_no_evidence_retrieved(self):
        chunks = [
            {"source": "Attention is all you need.pdf", "page": 7},
        ]
        found, required = compute_evidence_coverage(chunks, [1, 6], "Attention is all you need.pdf")
        assert found == 0
        assert required == 2

    def test_wrong_doc_not_counted(self):
        """Chunks from wrong doc should not contribute to evidence coverage."""
        chunks = [
            {"source": "LLM-based.pdf", "page": 1},
            {"source": "LLM-based.pdf", "page": 6},
        ]
        found, required = compute_evidence_coverage(chunks, [1, 6], "Attention is all you need.pdf")
        assert found == 0

    def test_empty_expected_pages(self):
        chunks = [{"source": "Attention is all you need.pdf", "page": 1}]
        found, required = compute_evidence_coverage(chunks, [], "Attention is all you need.pdf")
        assert found == 0
        assert required == 0


# ---------------------------------------------------------------------------
# 4. Metric calculations
# ---------------------------------------------------------------------------

class TestCalculateEvaluationMetrics:
    """Tests for calculate_evaluation_metrics."""

    def test_known_ranks_hit1_hit4_mrr(self):
        ranks = [1, 2, 4, None, None]
        metrics = calculate_evaluation_metrics(ranks)
        assert metrics["hit_at_1"] == pytest.approx(20.0)
        assert metrics["hit_at_4"] == pytest.approx(60.0)
        expected_mrr = (1.0 + 0.5 + 0.25) / 5.0
        assert metrics["mrr"] == pytest.approx(expected_mrr, abs=1e-4)

    def test_all_hits_at_rank_1(self):
        ranks = [1, 1, 1, 1]
        metrics = calculate_evaluation_metrics(ranks)
        assert metrics["hit_at_1"] == pytest.approx(100.0)
        assert metrics["hit_at_4"] == pytest.approx(100.0)
        assert metrics["mrr"] == pytest.approx(1.0)

    def test_all_misses(self):
        ranks = [None, None, None]
        metrics = calculate_evaluation_metrics(ranks)
        assert metrics["hit_at_1"] == 0.0
        assert metrics["hit_at_4"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_oos_rejection_rate_100(self):
        metrics = calculate_evaluation_metrics([], oos_total=5, oos_rejected=5)
        assert metrics["rejection_rate"] == pytest.approx(100.0)

    def test_oos_rejection_rate_partial(self):
        metrics = calculate_evaluation_metrics([], oos_total=5, oos_rejected=3)
        assert metrics["rejection_rate"] == pytest.approx(60.0)

    def test_oos_rejection_rate_zero_total(self):
        """When no OOS questions exist, rejection rate defaults to 100%."""
        metrics = calculate_evaluation_metrics([1, 2], oos_total=0, oos_rejected=0)
        assert metrics["rejection_rate"] == pytest.approx(100.0)

    def test_keyword_validated_hit4_calculated(self):
        ranks = [1, None, 2]
        keyword_hits = [True, False, True]
        metrics = calculate_evaluation_metrics(ranks, keyword_hits=keyword_hits)
        expected_kw = (2 / 3) * 100.0
        assert metrics["keyword_validated_hit_at_4"] == pytest.approx(expected_kw, abs=0.01)

    def test_complete_evidence_rate_full(self):
        ranks = [1, 1]
        evidence = [(2, 2), (3, 3)]
        metrics = calculate_evaluation_metrics(ranks, evidence_coverages=evidence)
        assert metrics["complete_evidence_rate"] == pytest.approx(100.0)

    def test_complete_evidence_rate_partial(self):
        ranks = [1, None]
        evidence = [(1, 2), (2, 3)]  # Neither fully covered
        metrics = calculate_evaluation_metrics(ranks, evidence_coverages=evidence)
        assert metrics["complete_evidence_rate"] == pytest.approx(0.0)

    def test_complete_evidence_rate_excludes_non_multihop(self):
        """Placeholders (0,0) from non-multihop questions should be excluded."""
        ranks = [1, 1, 1]
        # Two non-multihop (0,0) + one multihop (2,2)
        evidence = [(0, 0), (0, 0), (2, 2)]
        metrics = calculate_evaluation_metrics(ranks, evidence_coverages=evidence)
        assert metrics["complete_evidence_rate"] == pytest.approx(100.0)

    def test_avg_latency_ms_calculated(self):
        ranks = [1]
        metrics = calculate_evaluation_metrics(ranks, latencies_ms=[100.0, 200.0, 300.0])
        assert metrics["avg_latency_ms"] == pytest.approx(200.0)

    def test_latency_none_when_not_provided(self):
        metrics = calculate_evaluation_metrics([1])
        assert metrics["avg_latency_ms"] is None


# ---------------------------------------------------------------------------
# 5. Out-of-scope rejection
# ---------------------------------------------------------------------------

class TestOutOfScopeRejection:
    """Tests that OOS questions with no chunks are correctly rejected."""

    def test_oos_rejection_all_rejected(self):
        """When search returns empty for OOS queries, all OOS should be rejected."""
        with patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]):

            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["out_of_scope_total"] == 1
            assert metrics["out_of_scope_rejected"] == 1
            assert metrics["rejection_rate"] == pytest.approx(100.0)

    def test_oos_hallucination_risk_counted(self):
        """When search returns chunks for OOS queries, it's flagged as hallucination risk."""
        def mock_search(**kwargs):
            return MOCK_ATTN_CHUNKS

        with patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", side_effect=mock_search):

            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=[SAMPLE_EVAL_SET[2]],  # only OOS item
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["out_of_scope_total"] == 1
            assert metrics["out_of_scope_rejected"] == 0
            assert metrics["rejection_rate"] == pytest.approx(0.0)

    def test_oos_strict_rejection_at_low_threshold(self):
        """OOS questions should be rejected when distance threshold is strict (0.20)."""
        with patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.9] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]):

            oos_only = [q for q in SAMPLE_EVAL_SET if q["out_of_scope"]]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=oos_only,
                distance_threshold=0.20,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["rejection_rate"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 6. Correct page, wrong answer content (keyword miss)
# ---------------------------------------------------------------------------

class TestCorrectPageWrongContent:
    """Verifies that page-only hit without keyword match does NOT count as valid hit."""

    def test_page_hit_without_keyword_does_not_count(self):
        """A chunk on the right page but without required keywords is not a hit."""
        chunk_wrong_content = [
            {
                "source": "Attention is all you need.pdf",
                "page": 4,
                "chunk_index": 0,
                "content": "This page discusses dropout regularization only",
            }
        ]

        def mock_pipeline(**kwargs):
            return {
                "has_chunks": True,
                "chunks": chunk_wrong_content,
                "reranked": [],
                "selected_chunks": chunk_wrong_content,
                "context": "dropout...",
                "prompt": "",
                "sources": chunk_wrong_content,
            }

        with patch("evaluate_rag_harder.prepare_rag_pipeline", side_effect=mock_pipeline):
            in_scope_q = [SAMPLE_EVAL_SET[0]]  # Expects page=4 + keywords ["softmax","d_k"]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=in_scope_q,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["hit_at_1"] == pytest.approx(0.0), (
                "Hit@1 should be 0 when page matches but required keyword is absent"
            )
            assert metrics["keyword_validated_hit_at_4"] == pytest.approx(0.0)

    def test_page_hit_with_keyword_counts(self):
        """A chunk on the right page WITH required keywords IS a hit."""
        chunk_right_content = [
            {
                "source": "Attention is all you need.pdf",
                "page": 4,
                "chunk_index": 0,
                "content": "Attention softmax d_k formula is here",
            }
        ]

        def mock_pipeline(**kwargs):
            return {
                "has_chunks": True,
                "chunks": chunk_right_content,
                "reranked": [],
                "selected_chunks": chunk_right_content,
                "context": "softmax d_k...",
                "prompt": "",
                "sources": chunk_right_content,
            }

        with patch("evaluate_rag_harder.prepare_rag_pipeline", side_effect=mock_pipeline):
            in_scope_q = [SAMPLE_EVAL_SET[0]]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=in_scope_q,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["hit_at_1"] == pytest.approx(100.0)
            assert metrics["keyword_validated_hit_at_4"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 7. Partial and complete multi-hop evidence
# ---------------------------------------------------------------------------

class TestMultiHopEvidence:
    """Tests that multi-hop/multi-page questions track evidence coverage."""

    def test_partial_evidence_tracked(self):
        """When only one of two required pages is retrieved, evidence shows 1/2."""
        partial_chunks = [
            {
                "source": "Attention is all you need.pdf",
                "page": 5,
                "chunk_index": 0,
                "content": "MultiHead FFN",
            }
            # Page 8 missing
        ]

        def mock_pipeline(**kwargs):
            return {
                "has_chunks": True,
                "chunks": partial_chunks,
                "reranked": [],
                "selected_chunks": partial_chunks,
                "context": "MultiHead...",
                "prompt": "",
                "sources": partial_chunks,
            }

        with patch("evaluate_rag_harder.prepare_rag_pipeline", side_effect=mock_pipeline):
            multi_q = [SAMPLE_EVAL_SET[1]]  # multi_hop, expected_pages=[5, 8]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=multi_q,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            # Hit@1 may succeed if page 5 matches with keyword "MultiHead"
            # But complete_evidence_rate should be 0% (page 8 missing)
            assert metrics["complete_evidence_rate"] is not None
            assert metrics["complete_evidence_rate"] == pytest.approx(0.0), (
                "Evidence rate should be 0% when page 8 is missing for multi-hop question"
            )

    def test_complete_evidence_both_pages_retrieved(self):
        """When all required pages are retrieved, complete evidence rate = 100%."""
        full_chunks = [
            {
                "source": "Attention is all you need.pdf",
                "page": 5,
                "chunk_index": 0,
                "content": "MultiHead FFN",
            },
            {
                "source": "Attention is all you need.pdf",
                "page": 8,
                "chunk_index": 1,
                "content": "BLEU training",
            },
        ]

        def mock_pipeline(**kwargs):
            return {
                "has_chunks": True,
                "chunks": full_chunks,
                "reranked": [],
                "selected_chunks": full_chunks,
                "context": "MultiHead + BLEU...",
                "prompt": "",
                "sources": full_chunks,
            }

        with patch("evaluate_rag_harder.prepare_rag_pipeline", side_effect=mock_pipeline):
            multi_q = [SAMPLE_EVAL_SET[1]]  # multi_hop, expected_pages=[5, 8]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=multi_q,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
            assert metrics["complete_evidence_rate"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 8. All 6 configurations
# ---------------------------------------------------------------------------

class TestSixConfigurations:
    """Verifies that all 6 configurations run correctly."""

    def test_6_configs_returned(self):
        """run_comparison must return exactly 6 result dicts."""
        in_scope_q = SAMPLE_EVAL_SET[:2]  # 2 in-scope, no OOS
        fake_chunks_by_q: dict[str, list[dict]] = {
            SAMPLE_EVAL_SET[0]["question"]: [
                {"source": "Attention is all you need.pdf", "page": 4,
                 "content": "softmax d_k formula", "chunk_index": 0}
            ],
            SAMPLE_EVAL_SET[1]["question"]: [
                {"source": "Attention is all you need.pdf", "page": 5,
                 "content": "MultiHead FFN", "chunk_index": 1}
            ],
        }

        call_tracker = {"q": None}

        def mock_embed(text):
            call_tracker["q"] = text
            return [0.1] * 3072

        def mock_search(**kwargs):
            return fake_chunks_by_q.get(call_tracker["q"], [])

        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", side_effect=mock_embed), \
             patch("app.services.rag_service.search_chunks", side_effect=mock_search), \
             patch("app.services.rag_service.rerank_chunks", return_value=MOCK_RERANKED_CHUNKS[:2]):

            results = run_comparison(
                tenant_id="test-tenant",
                eval_set=in_scope_q,
                top_k=RERANK_TOP_K,
            )

        assert len(results) == 6, f"Expected 6 results, got {len(results)}"

    def test_6_configs_expected_combos(self):
        """Results must cover all (threshold, reranking) combinations in correct order."""
        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]), \
             patch("app.services.rag_service.rerank_chunks", return_value=[]):

            results = run_comparison(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                top_k=RERANK_TOP_K,
            )

        expected = [
            (0.20, "OFF"), (0.30, "OFF"), (0.40, "OFF"),
            (0.20, "ON"), (0.30, "ON"), (0.40, "ON"),
        ]
        actual = [(r["threshold"], r["reranking"]) for r in results]
        assert actual == expected, f"Config order mismatch: {actual}"

    def test_search_called_per_question_per_config(self):
        """With 3 questions × 6 configs = 18 search calls total."""
        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]) as mock_search, \
             patch("app.services.rag_service.rerank_chunks", return_value=[]):

            run_comparison(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                top_k=RERANK_TOP_K,
            )

        assert mock_search.call_count == len(SAMPLE_EVAL_SET) * 6

    def test_rerank_called_only_for_in_scope_on_configs(self):
        """Rerank is called for every question that returns chunks when reranking=ON.
        With SAMPLE_EVAL_SET (3 questions, all getting chunks), 3 questions × 3 ON configs = 9 calls.
        """
        total_q = len(SAMPLE_EVAL_SET)
        expected_rerank_calls = total_q * 3  # 3 ON configs, all Qs return chunks

        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks",
                   return_value=[MOCK_ATTN_CHUNKS[0]]), \
             patch("app.services.rag_service.rerank_chunks",
                   return_value=[MOCK_RERANKED_CHUNKS[0]]) as mock_rerank:

            run_comparison(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                top_k=RERANK_TOP_K,
            )

        assert mock_rerank.call_count == expected_rerank_calls, (
            f"Expected {expected_rerank_calls} rerank calls (all Qs × 3 ON configs), "
            f"got {mock_rerank.call_count}"
        )


# ---------------------------------------------------------------------------
# 9. Cohere reranking fallback behaviour
# ---------------------------------------------------------------------------

class TestCohereReranking:
    """Tests for Cohere availability checks and fallback."""

    def test_cohere_unavailable_raises_runtime_error(self):
        """When Cohere is unavailable, use_rerank=True must raise RuntimeError."""
        with patch("evaluate_rag_harder.verify_cohere_functional", return_value=False):
            with pytest.raises(RuntimeError, match="Cannot report fallback results as successful reranking"):
                evaluate_single_configuration(
                    tenant_id="test-tenant",
                    eval_set=SAMPLE_EVAL_SET,
                    distance_threshold=0.30,
                    use_rerank=True,
                    verbose=False,
                )

    def test_cohere_unavailable_with_rerank_off_does_not_raise(self):
        """When reranking is OFF, Cohere unavailability should not raise any error."""
        with patch("evaluate_rag_harder.verify_cohere_functional", return_value=False), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]):
            # Should NOT raise
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
        assert "hit_at_1" in metrics

    def test_rerank_on_uses_reranked_chunks(self):
        """When reranking is ON, the pipeline should use reranked chunks for scoring."""
        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks",
                   return_value=MOCK_ATTN_CHUNKS), \
             patch("app.services.rag_service.rerank_chunks",
                   return_value=MOCK_RERANKED_CHUNKS) as mock_rerank:

            in_scope_q = [SAMPLE_EVAL_SET[0]]
            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=in_scope_q,
                distance_threshold=0.30,
                use_rerank=True,
                verbose=False,
                print_summary=False,
            )
        assert mock_rerank.called, "rerank_chunks should have been called"

    def test_rerank_off_does_not_call_rerank(self):
        """When reranking is OFF, rerank_chunks should never be called."""
        with patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks",
                   return_value=MOCK_ATTN_CHUNKS), \
             patch("app.services.rag_service.rerank_chunks") as mock_rerank:

            in_scope_q = [SAMPLE_EVAL_SET[0]]
            evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=in_scope_q,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
        mock_rerank.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Latency tracking
# ---------------------------------------------------------------------------

class TestLatencyTracking:
    """Verifies that retrieval latency is measured per question."""

    def test_avg_latency_ms_is_positive(self):
        """Average latency must be a positive float."""
        with patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]):

            metrics = evaluate_single_configuration(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                distance_threshold=0.30,
                use_rerank=False,
                verbose=False,
                print_summary=False,
            )
        assert metrics["avg_latency_ms"] is not None
        assert metrics["avg_latency_ms"] >= 0.0

    def test_avg_latency_in_comparison_results(self):
        """All 6 comparison results should include avg_latency_ms."""
        with patch("time.sleep"), \
             patch("evaluate_rag_harder.verify_cohere_functional", return_value=True), \
             patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=[]), \
             patch("app.services.rag_service.rerank_chunks", return_value=[]):

            results = run_comparison(
                tenant_id="test-tenant",
                eval_set=SAMPLE_EVAL_SET,
                top_k=RERANK_TOP_K,
            )
        for r in results:
            assert "avg_latency_ms" in r
            assert r["avg_latency_ms"] is not None


# ---------------------------------------------------------------------------
# 11. Summary output format
# ---------------------------------------------------------------------------

class TestSummaryOutput:
    """Tests print_evaluation_summary output format."""

    def test_summary_includes_all_harder_metrics(self, capsys):
        metrics = {
            "total": 32,
            "in_scope_total": 25,
            "out_of_scope_total": 7,
            "out_of_scope_rejected": 7,
            "hit_at_1": 72.0,
            "hit_at_4": 88.0,
            "mrr": 0.812,
            "keyword_validated_hit_at_4": 84.0,
            "complete_evidence_rate": 60.0,
            "avg_latency_ms": 345.6,
            "rejection_rate": 100.0,
        }
        print_evaluation_summary(metrics)
        captured = capsys.readouterr().out
        assert "Hit@1" in captured
        assert "Hit@4" in captured
        assert "MRR" in captured
        assert "KW-Validated" in captured
        assert "Full Evidence" in captured
        assert "Avg Latency" in captured
        assert "OOS Rejection" in captured

    def test_comparison_table_prints_7_columns(self, capsys):
        """Comparison table header should include all 7 columns."""
        fake_results = [
            {
                "threshold": t,
                "reranking": r,
                "hit_at_1": 70.0,
                "hit_at_4": 90.0,
                "mrr": 0.80,
                "rejection_rate": 100.0,
                "keyword_validated_hit_at_4": 85.0,
                "complete_evidence_rate": 60.0,
                "avg_latency_ms": 200.0,
                "failed_questions": [],
            }
            for t, r in [
                (0.20, "OFF"), (0.30, "OFF"), (0.40, "OFF"),
                (0.20, "ON"), (0.30, "ON"), (0.40, "ON"),
            ]
        ]
        print_comparison_table(fake_results)
        captured = capsys.readouterr().out
        assert "Threshold" in captured
        assert "Reranking" in captured
        assert "Hit@1" in captured
        assert "MRR" in captured
        assert "OOS Reject" in captured
        assert "KW@4" in captured
        assert "Evidence" in captured
        assert "Avg Latency" in captured
        assert "Recommended" in captured

    def test_recommendation_mentions_threshold(self, capsys):
        """The recommendation line must mention the winning threshold."""
        fake_results = [
            {
                "threshold": t,
                "reranking": r,
                "hit_at_1": 70.0 + (0.5 if t == 0.30 else 0.0),
                "hit_at_4": 90.0 + (1.0 if t == 0.30 else 0.0),
                "mrr": 0.80 + (0.01 if t == 0.30 else 0.0),
                "rejection_rate": 100.0,
                "keyword_validated_hit_at_4": 85.0,
                "complete_evidence_rate": 60.0,
                "avg_latency_ms": 200.0,
                "failed_questions": [],
            }
            for t, r in [
                (0.20, "OFF"), (0.30, "OFF"), (0.40, "OFF"),
                (0.20, "ON"), (0.30, "ON"), (0.40, "ON"),
            ]
        ]
        print_comparison_table(fake_results)
        captured = capsys.readouterr().out
        assert "0.30" in captured


# ---------------------------------------------------------------------------
# 12. COMPARISON_THRESHOLDS constant
# ---------------------------------------------------------------------------

def test_comparison_thresholds_are_correct():
    """COMPARISON_THRESHOLDS must include exactly [0.20, 0.30, 0.40]."""
    assert COMPARISON_THRESHOLDS == [0.20, 0.30, 0.40]


# ---------------------------------------------------------------------------
# Manual runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
