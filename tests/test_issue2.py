"""Test Issue 2: Remove Evaluation Code from Production Request Path.

Verifies:
1. _check_and_log_benchmark_evaluation is deleted and inspect is not imported/used in rag_service.
2. Production requests to /ask and /ask-stream do not execute evaluation benchmark matching.
3. /ask and /ask-stream work correctly for normal RAG queries.
4. evaluate_rag.py calculates Hit@1, Hit@4, MRR, and out-of-scope rejection rate accurately.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_user
import app.services.rag_service as rag_service
from evaluate_rag import calculate_evaluation_metrics


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleared after each test."""
    yield
    app.dependency_overrides.clear()


def test_eval_function_removed_from_rag_service():
    """Verify _check_and_log_benchmark_evaluation function no longer exists."""
    assert not hasattr(rag_service, "_check_and_log_benchmark_evaluation"), (
        "_check_and_log_benchmark_evaluation must be completely deleted from rag_service"
    )


def test_ask_and_stream_endpoints_work_for_rag_queries():
    """Verify /ask and /ask-stream return correct 200 responses with RAG pipeline."""
    mock_user = {"uid": "user_issue2", "email": "issue2@example.com"}
    mock_record = {
        "uid": "user_issue2",
        "email": "issue2@example.com",
        "tenant_id": "tenant_issue2",
        "permissions": ["READ", "WRITE"],
        "is_active": True,
    }

    app.dependency_overrides[get_current_user] = lambda: mock_user

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = "The Transformer model uses multi-head self-attention."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_resp

    mock_stream_chunk = MagicMock()
    mock_stream_chunk.text = "The Transformer model."
    mock_client.models.generate_content_stream.return_value = [mock_stream_chunk]

    mock_chunks = [
        {"source": "paper.pdf", "page": 1, "chunk_index": 0, "content": "Transformer architecture."}
    ]

    with patch("app.dependencies.resolve_authorized_user", return_value=(mock_record, "tenant_issue2", None)), \
         patch("app.routers.chat.resolve_authorized_user", return_value=(mock_record, "tenant_issue2", None)), \
         patch("app.services.rag_service.get_gemini_client", return_value=mock_client), \
         patch("app.services.rag_service.embed_query", return_value=[0.05] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=mock_chunks), \
         patch("app.services.rag_service.rerank_chunks", return_value=mock_chunks):

        with TestClient(app) as client:
            # 1. Test /ask
            res = client.post("/ask", json={"question": "What is the Transformer architecture?"})
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert "answer" in data
            assert "Transformer" in data["answer"]
            assert len(data["sources"]) == 1
            assert data["sources"][0]["source"] == "paper.pdf"

            # 2. Test /ask-stream
            stream_res = client.post("/ask-stream", json={"question": "What is the Transformer architecture?"})
            assert stream_res.status_code == 200
            assert "text/event-stream" in stream_res.headers.get("content-type", "")
            assert "data:" in stream_res.text


def test_normal_request_does_not_execute_eval_matching(capsys):
    """Verify normal requests do not print [Benchmark Match] logs or inspect stack."""
    mock_chunks = [
        {"source": "test.pdf", "page": 1, "chunk_index": 0, "content": "Sample content"}
    ]

    with patch("app.services.rag_service.embed_query", return_value=[0.01] * 3072), \
         patch("app.services.rag_service.search_chunks", return_value=mock_chunks), \
         patch("app.services.rag_service.rerank_chunks", return_value=mock_chunks):

        # Execute prepare_rag_pipeline with a question that exists in EVAL_SET
        result = rag_service.prepare_rag_pipeline(
            question="What is the primary purpose of the Transformer model?",
            tenant_id="tenant_issue2",
        )

        assert result["has_chunks"] is True
        captured = capsys.readouterr()
        # Verify no benchmark match log was emitted
        assert "[Benchmark Match]" not in captured.out
        assert "Hit Rank" not in captured.out


def test_evaluate_rag_metric_calculations():
    """Verify evaluate_rag.py calculates Hit@1, Hit@4, MRR, and out-of-scope rejection rate accurately."""
    # Scenario:
    # 4 in-scope questions:
    # Q1: Hit at rank 1 -> Reciprocal rank = 1.0
    # Q2: Hit at rank 2 -> Reciprocal rank = 0.5 (within top 4)
    # Q3: Hit at rank 4 -> Reciprocal rank = 0.25 (within top 4)
    # Q4: Miss (None)   -> Reciprocal rank = 0.0
    #
    # Expected metrics:
    # Hit@1 = 1 / 4 = 25.0%
    # Hit@4 = 3 / 4 = 75.0%
    # MRR   = (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 0.4375
    #
    # 2 out-of-scope questions:
    # 2 rejected -> Rejection rate = 100.0%
    first_ranks = [1, 2, 4, None]
    oos_total = 2
    oos_rejected = 2

    metrics = calculate_evaluation_metrics(
        first_ranks=first_ranks,
        oos_total=oos_total,
        oos_rejected=oos_rejected,
    )

    assert metrics["in_scope_total"] == 4
    assert metrics["total"] == 6
    assert metrics["hit_at_1"] == pytest.approx(25.0)
    assert metrics["hit_at_4"] == pytest.approx(75.0)
    assert metrics["mrr"] == pytest.approx(0.4375)
    assert metrics["out_of_scope_total"] == 2
    assert metrics["out_of_scope_rejected"] == 2
    assert metrics["rejection_rate"] == pytest.approx(100.0)
