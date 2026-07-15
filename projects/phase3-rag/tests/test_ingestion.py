from __future__ import annotations

from ingestion import chunk_by_fixed_size, chunk_recursive, enrich_metadata
from retrieval import hybrid_rrf_fusion


def test_fixed_chunking_produces_chunks() -> None:
    text = "x" * 2000
    chunks = chunk_by_fixed_size(text, chunk_size=512, overlap=50)
    assert len(chunks) > 1


def test_recursive_chunking() -> None:
    long_text = " ".join(["word"] * 1000)
    chunk_size = 512
    chunks = chunk_recursive(long_text, chunk_size=chunk_size)
    assert all(len(c) <= chunk_size * 1.5 for c in chunks)


def test_metadata_enrichment() -> None:
    chunks = ["first chunk", "second chunk", "third chunk"]
    result = enrich_metadata(chunks, source="test.txt", section="intro")
    assert all("source" in item for item in result)
    assert all("chunk_index" in item for item in result)
    assert result[0]["source"] == "test.txt"
    assert result[1]["chunk_index"] == 1


def test_rrf_fusion_scores_first_place_highest() -> None:
    dense = ["doc_A", "doc_B", "doc_C"]
    sparse = ["doc_A", "doc_D", "doc_E"]
    fused = hybrid_rrf_fusion(dense, sparse)
    assert fused[0] == "doc_A"
