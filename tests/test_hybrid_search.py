"""
tests/test_hybrid_search.py

Pytest suite for Feature 2: Hybrid Search (BM25 + Vector).
Covers:
  - Config defaults and env overrides (SEARCH_MODE, HYBRID_ALPHA, HYBRID_SCORE_THRESHOLD)
  - search_chunks in vector mode vs hybrid mode (alpha, max_vector_distance, MetadataQuery)
  - Hybrid fusion score handling (cosine distance threshold NOT applied to fusion scores)
  - Optional hybrid_score_threshold filtering
  - Invalid search_mode validation
  - Empty query_text fallback to near_vector
  - prepare_rag_pipeline preserving 20-candidate retrieval, top-4 Cohere rerank, tenant isolation
  - evaluate_rag_harder.py hybrid evaluation integration
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import DISTANCE_THRESHOLD, HYBRID_ALPHA, HYBRID_SCORE_THRESHOLD, RERANK_TOP_K, RETRIEVAL_TOP_K, SEARCH_MODE
from app.services.rag_service import prepare_rag_pipeline
from app.services.vector_store import search_chunks
from evaluate_rag_harder import HYBRID_ALPHAS, evaluate_single_configuration


def _make_weaviate_obj(content: str, source: str = "Attention is all you need.pdf", page: int = 4,
                       chunk_index: int = 0, distance: float | None = None, score: float | None = None):
    return SimpleNamespace(
        properties={
            "content": content,
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "doc_id": "doc-1",
        },
        metadata=SimpleNamespace(
            distance=distance,
            certainty=None,
            score=score,
            explain_score="hybrid" if score is not None else None,
        ),
    )


def _mock_weaviate_client(objects):
    tenant_col = MagicMock()
    tenant_col.query.hybrid.return_value = SimpleNamespace(objects=objects)
    tenant_col.query.near_vector.return_value = SimpleNamespace(objects=objects)

    col = MagicMock()
    col.with_tenant.return_value = tenant_col

    client = MagicMock()
    client.collections.exists.return_value = True
    client.collections.get.return_value = col
    return client, tenant_col


class TestHybridConfig:
    def test_hybrid_alphas_to_evaluate(self):
        assert HYBRID_ALPHAS == [0.25, 0.50, 0.75]

    def test_default_config_values_valid(self):
        assert SEARCH_MODE in ("vector", "hybrid")
        assert 0.0 <= HYBRID_ALPHA <= 1.0
        assert HYBRID_SCORE_THRESHOLD >= 0.0


class TestVectorStoreHybridSearch:
    def test_hybrid_mode_calls_native_weaviate_hybrid(self):
        objs = [_make_weaviate_obj("Scaled Dot-Product Attention softmax", score=0.82, distance=None)]
        client, tenant_col = _mock_weaviate_client(objs)

        with patch("app.services.vector_store.get_weaviate_client", return_value=client), \
             patch("app.services.vector_store.ensure_tenant", return_value="tenant-a"):
            results = search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                limit=20,
                distance_threshold=0.40,
                query_text="Scaled Dot-Product Attention",
                search_mode="hybrid",
                alpha=0.50,
            )

        tenant_col.query.hybrid.assert_called_once()
        tenant_col.query.near_vector.assert_not_called()
        call_kwargs = tenant_col.query.hybrid.call_args.kwargs
        assert call_kwargs["query"] == "Scaled Dot-Product Attention"
        assert call_kwargs["alpha"] == 0.50
        assert call_kwargs["limit"] == 20
        assert call_kwargs["max_vector_distance"] == 0.40
        assert len(results) == 1
        assert results[0]["score"] == 0.82
        assert results[0]["tenant"] == "tenant-a"

    def test_hybrid_does_not_apply_cosine_distance_threshold_to_fusion_score(self):
        """Fusion score 0.15 is < 0.30, and score 0.90 is > 0.30 — neither should be treated as cosine distance."""
        objs = [
            _make_weaviate_obj("Chunk A", score=0.90, distance=None),
            _make_weaviate_obj("Chunk B", score=0.15, distance=None),
        ]
        client, _ = _mock_weaviate_client(objs)

        with patch("app.services.vector_store.get_weaviate_client", return_value=client), \
             patch("app.services.vector_store.ensure_tenant", return_value="tenant-a"):
            results = search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                limit=20,
                distance_threshold=0.30,
                query_text="test query",
                search_mode="hybrid",
                alpha=0.25,
                hybrid_score_threshold=0.0,
            )

        assert len(results) == 2

    def test_hybrid_score_threshold_filters_low_fusion_scores_when_configured(self):
        objs = [
            _make_weaviate_obj("High fusion", score=0.75, distance=None),
            _make_weaviate_obj("Low fusion", score=0.10, distance=None),
        ]
        client, _ = _mock_weaviate_client(objs)

        with patch("app.services.vector_store.get_weaviate_client", return_value=client), \
             patch("app.services.vector_store.ensure_tenant", return_value="tenant-a"):
            results = search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                limit=20,
                distance_threshold=0.40,
                query_text="test query",
                search_mode="hybrid",
                alpha=0.75,
                hybrid_score_threshold=0.30,
            )

        assert len(results) == 1
        assert results[0]["content"] == "High fusion"

    def test_vector_mode_preserves_cosine_distance_filtering(self):
        objs = [
            _make_weaviate_obj("Close chunk", distance=0.22),
            _make_weaviate_obj("Far chunk", distance=0.55),
        ]
        client, tenant_col = _mock_weaviate_client(objs)

        with patch("app.services.vector_store.get_weaviate_client", return_value=client), \
             patch("app.services.vector_store.ensure_tenant", return_value="tenant-a"):
            results = search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                limit=20,
                distance_threshold=0.30,
                search_mode="vector",
            )

        tenant_col.query.near_vector.assert_called_once()
        tenant_col.query.hybrid.assert_not_called()
        assert len(results) == 1
        assert results[0]["content"] == "Close chunk"

    def test_invalid_search_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported search_mode"):
            search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                search_mode="bm25_only",
            )

    def test_hybrid_without_query_text_falls_back_to_near_vector(self):
        objs = [_make_weaviate_obj("Chunk", distance=0.20)]
        client, tenant_col = _mock_weaviate_client(objs)

        with patch("app.services.vector_store.get_weaviate_client", return_value=client), \
             patch("app.services.vector_store.ensure_tenant", return_value="tenant-a"):
            results = search_chunks(
                query_vector=[0.1] * 3072,
                tenant_id="tenant-a",
                query_text=None,
                search_mode="hybrid",
            )

        tenant_col.query.near_vector.assert_called_once()
        assert len(results) == 1


class TestPipelineAndEvalHybridIntegration:
    def test_prepare_rag_pipeline_hybrid_passes_20_candidates_and_reranks_top_4(self):
        candidates = [
            {"source": "Attention is all you need.pdf", "page": 4, "chunk_index": i,
             "content": f"Chunk {i}", "score": 0.9 - i * 0.01}
            for i in range(20)
        ]
        reranked = candidates[:4]

        with patch("app.services.rag_service.embed_query", return_value=[0.1] * 3072), \
             patch("app.services.rag_service.search_chunks", return_value=candidates) as mock_search, \
             patch("app.services.rag_service.rerank_chunks", return_value=reranked) as mock_rerank:
            res = prepare_rag_pipeline(
                question="What is Scaled Dot-Product Attention?",
                tenant_id="tenant-xyz",
                use_rerank=True,
                top_k=4,
                distance_threshold=0.40,
                search_mode="hybrid",
                alpha=0.50,
            )

        mock_search.assert_called_once()
        sk = mock_search.call_args.kwargs
        assert sk["limit"] == RETRIEVAL_TOP_K == 20
        assert sk["tenant_id"] == "tenant-xyz"
        assert sk["search_mode"] == "hybrid"
        assert sk["alpha"] == 0.50
        assert sk["query_text"] == "What is Scaled Dot-Product Attention?"
        mock_rerank.assert_called_once()
        assert mock_rerank.call_args.kwargs["top_k"] == RERANK_TOP_K == 4
        assert len(res["selected_chunks"]) == 4

    def test_evaluate_single_configuration_records_hybrid_metadata(self):
        eval_item = [{
            "id": 15,
            "type": "paraphrase",
            "question": "Suppress activations dropout label smoothing",
            "expected_doc": "Attention is all you need.pdf",
            "expected_pages": [8],
            "required_keywords": ["dropout", "label smoothing"],
            "out_of_scope": False,
        }]
        mock_pipeline_res = {
            "has_chunks": True,
            "chunks": [{"source": "Attention is all you need.pdf", "page": 8, "chunk_index": 0, "content": "dropout 0.1 label smoothing 0.1"}],
            "reranked": [],
            "selected_chunks": [{"source": "Attention is all you need.pdf", "page": 8, "chunk_index": 0, "content": "dropout 0.1 label smoothing 0.1"}],
            "context": "...",
            "prompt": "...",
            "sources": [],
        }

        with patch("evaluate_rag_harder.prepare_rag_pipeline", return_value=mock_pipeline_res) as mock_pipe:
            metrics = evaluate_single_configuration(
                tenant_id="tenant-xyz",
                eval_set=eval_item,
                distance_threshold=0.40,
                use_rerank=False,
                verbose=False,
                print_summary=False,
                search_mode="hybrid",
                alpha=0.25,
            )

        assert metrics["search_mode"] == "hybrid"
        assert metrics["alpha"] == 0.25
        assert metrics["hit_at_1"] == 100.0
        assert mock_pipe.call_args.kwargs["search_mode"] == "hybrid"
        assert mock_pipe.call_args.kwargs["alpha"] == 0.25
