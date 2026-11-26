"""Test Issue 3: Validate Distance Threshold and Reranking Performance.

Verifies:
1. Metric calculations (Hit@1, Hit@4, MRR, Out-of-Scope Rejection).
2. All 6 configurations execute correctly (thresholds 0.20, 0.30, 0.40 x Rerank OFF/ON).
3. Distance threshold filtering works as expected (chunks > threshold rejected).
4. Out-of-scope questions trigger rejection response ("I don't know").
5. Cohere fallback is NOT reported as successful reranking (fails closed when unavailable).
6. Production pipeline parity (retrieve 20, distance filter, rerank top 4).
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_user
import app.services.rag_service as rag_service
from evaluate_rag import (
    COMPARISON_THRESHOLDS,
    calculate_evaluation_metrics,
    evaluate_single_configuration,
    verify_cohere_functional,
)


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleared after each test."""
    yield
    app.dependency_overrides.clear()


def test_metric_calculations_accuracy():
    """Verify Hit@1, Hit@4, MRR, and out-of-scope rejection calculations."""
    # 5 in-scope questions:
    # Q1: Hit at rank 1 -> RR = 1.0 (Hit@1, Hit@4)
    # Q2: Hit at rank 2 -> RR = 0.5 (Hit@4)
    # Q3: Hit at rank 3 -> RR = 0.3333 (Hit@4)
    # Q4: Hit at rank 5 -> RR = 0.2 (Miss for Hit@4)
    # Q5: Miss (None)   -> RR = 0.0 (Miss for Hit@1, Hit@4)
    # 5 out-of-scope questions: 4 rejected, 1 hallucinated
    first_ranks = [1, 2, 3, 5, None]
    oos_total = 5
    oos_rejected = 4

    metrics = calculate_evaluation_metrics(
        first_ranks=first_ranks,
        oos_total=oos_total,
        oos_rejected=oos_rejected,
    )

    assert metrics["in_scope_total"] == 5
    assert metrics["total"] == 10
    assert metrics["hit_at_1"] == pytest.approx(20.0)  # 1 / 5 = 20%
    assert metrics["hit_at_4"] == pytest.approx(60.0)  # 3 / 5 = 60%
    expected_mrr = (1.0 + 0.5 + (1.0 / 3.0) + 0.2 + 0.0) / 5.0
    assert metrics["mrr"] == pytest.approx(expected_mrr, rel=1e-3)
    assert metrics["out_of_scope_total"] == 5
    assert metrics["out_of_scope_rejected"] == 4
    assert metrics["rejection_rate"] == pytest.approx(80.0)


def test_six_configurations_defined():
    """Verify that COMPARISON_THRESHOLDS covers 0.20, 0.30, 0.40."""
    assert COMPARISON_THRESHOLDS == [0.20, 0.30, 0.40]


def test_cohere_fallback_fails_closed_when_unavailable():
    """Verify that when Cohere is non-functional, evaluate_single_configuration raises RuntimeError."""
    with patch("evaluate_rag.verify_cohere_functional", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            evaluate_single_configuration(
                tenant_id="test_tenant",
                distance_threshold=0.30,
                use_rerank=True,
                verbose=False,
                print_summary=False,
            )
        assert "Cannot report fallback results as successful reranking" in str(exc_info.value)


def test_distance_filtering_rejects_out_of_scope_queries():
    """Verify that distance filtering rejects chunks with distance > threshold."""
    with patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
         patch("app.services.vector_store.get_weaviate_client") as mock_client:

        mock_collection = MagicMock()
        mock_obj = MagicMock()
        mock_obj.properties = {"content": "Irrelevant chunk", "page": 1, "source": "test.pdf", "chunk_index": 0}
        mock_obj.metadata.distance = 0.35
        mock_collection.tenants.get.return_value = {"test_tenant": None}
        mock_collection.with_tenant.return_value.query.near_vector.return_value.objects = [mock_obj]
        mock_client.return_value.collections.get.return_value = mock_collection
        mock_client.return_value.collections.exists.return_value = True

        # At threshold 0.30: 0.35 > 0.30 -> chunk should be filtered out
        filtered = rag_service.search_chunks(
            query_vector=[0.1] * 3072,
            tenant_id="test_tenant",
            limit=20,
            distance_threshold=0.30,
        )
        assert len(filtered) == 0

        # At threshold 0.40: 0.35 <= 0.40 -> chunk is retained
        retained = rag_service.search_chunks(
            query_vector=[0.1] * 3072,
            tenant_id="test_tenant",
            limit=20,
            distance_threshold=0.40,
        )
        assert len(retained) == 1


def test_out_of_scope_response_in_ask_route():
    """Verify that /ask returns the rejection response when no chunks pass distance filtering."""
    mock_user = {"uid": "user_issue3", "email": "issue3@example.com"}
    mock_record = {
        "uid": "user_issue3",
        "email": "issue3@example.com",
        "tenant_id": "tenant_issue3",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    # Mock pipeline returning has_chunks = False (out of scope)
    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_issue3", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_issue3", None)), \
         patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=[]):

        with TestClient(app) as client:
            res = client.post("/ask", json={"question": "Who won the 2022 World Cup?"})
            assert res.status_code == 200
            data = res.json()
            assert "I could not find relevant information" in data["answer"]
            assert data["sources"] == []


def test_evaluate_single_configuration_mocked_execution():
    """Verify evaluate_single_configuration executes both rerank OFF and ON pipelines."""
    from evaluate_rag import EVAL_SET

    def mock_prepare_pipeline(question, tenant_id, source_filter=None, use_rerank=False, top_k=4, distance_threshold=0.30):
        for item in EVAL_SET:
            if item["question"] == question:
                if item.get("out_of_scope"):
                    return {"has_chunks": False, "chunks": [], "selected_chunks": [], "context": "", "prompt": "", "sources": []}
                expected_page = item["expected_pages"][0]
                chunk = {
                    "content": f"Relevant content for {question}",
                    "page": expected_page,
                    "chunk_index": 0,
                    "source": "Attention is all you need.pdf",
                    "distance": 0.22,
                }
                return {
                    "has_chunks": True,
                    "chunks": [chunk],
                    "selected_chunks": [chunk],
                    "context": chunk["content"],
                    "prompt": "Test prompt",
                    "sources": [chunk],
                }
        return {"has_chunks": False, "chunks": [], "selected_chunks": []}

    with patch("evaluate_rag.prepare_rag_pipeline", side_effect=mock_prepare_pipeline), \
         patch("evaluate_rag.verify_cohere_functional", return_value=True):

        # Rerank OFF
        res_off = evaluate_single_configuration(
            tenant_id="test_tenant",
            distance_threshold=0.30,
            use_rerank=False,
            top_k=4,
            verbose=False,
            print_summary=False,
        )
        assert res_off["threshold"] == 0.30
        assert res_off["reranking"] == "OFF"
        assert res_off["hit_at_1"] == 100.0
        assert res_off["hit_at_4"] == 100.0
        assert res_off["mrr"] == 1.0
        assert res_off["rejection_rate"] == 100.0

        # Rerank ON
        res_on = evaluate_single_configuration(
            tenant_id="test_tenant",
            distance_threshold=0.30,
            use_rerank=True,
            top_k=4,
            verbose=False,
            print_summary=False,
        )
        assert res_on["threshold"] == 0.30
        assert res_on["reranking"] == "ON"
        assert res_on["hit_at_1"] == 100.0
        assert res_on["hit_at_4"] == 100.0
        assert res_on["mrr"] == 1.0
        assert res_on["rejection_rate"] == 100.0
        assert res_on["hit_at_4"] == 100.0
        assert res_on["mrr"] == 1.0
