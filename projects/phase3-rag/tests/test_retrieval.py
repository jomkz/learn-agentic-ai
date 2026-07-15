from __future__ import annotations

import pytest
from pydantic import ValidationError
from retrieval import RetrievalResult, VectorStoreConfig, hybrid_rrf_fusion


def test_vector_store_config_defaults() -> None:
    cfg = VectorStoreConfig(store_type="pgvector")
    assert cfg.collection_name == "documents"


def test_vector_store_config_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        VectorStoreConfig(store_type="pinecone")  # type: ignore[arg-type]


def test_retrieval_result_model() -> None:
    result = RetrievalResult(content="x", score=0.9, metadata={})
    assert result.content == "x"
    assert result.score == pytest.approx(0.9)
    assert result.metadata == {}


def test_rrf_fusion_empty_lists() -> None:
    assert hybrid_rrf_fusion([], []) == []


def test_rrf_fusion_deduplicates() -> None:
    dense = ["doc_A", "doc_B"]
    sparse = ["doc_A", "doc_C"]
    result = hybrid_rrf_fusion(dense, sparse)
    assert result.count("doc_A") == 1


def test_rrf_fusion_ordering() -> None:
    # doc_A is rank-0 in both lists — it must accumulate the highest RRF score
    dense = ["doc_A", "doc_B", "doc_C"]
    sparse = ["doc_A", "doc_D", "doc_E"]
    result = hybrid_rrf_fusion(dense, sparse)
    assert result[0] == "doc_A"


def test_rrf_fusion_only_dense() -> None:
    result = hybrid_rrf_fusion(["a", "b"], [])
    assert result == ["a", "b"]


def test_rrf_fusion_respects_k() -> None:
    # "A" is rank-0 in dense only; "B" is rank-5 in both lists.
    # With small k the rank-0 advantage dominates → A before B.
    # With large k appearing-in-both dominates → B before A.
    dense = ["A", "p1", "p2", "p3", "p4", "B"]
    sparse = ["s1", "s2", "s3", "s4", "s5", "B"]

    result_small_k = hybrid_rrf_fusion(dense, sparse, k=3)
    result_large_k = hybrid_rrf_fusion(dense, sparse, k=5)

    # Verify k visibly changes relative ordering of A vs B
    assert result_small_k.index("A") < result_small_k.index("B")
    assert result_large_k.index("B") < result_large_k.index("A")
