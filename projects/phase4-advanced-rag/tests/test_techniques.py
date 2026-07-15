from __future__ import annotations

from techniques import TechniqueResult, hyde_retrieval, multi_query_retrieval, rerank_with_scores


def test_technique_result_model() -> None:
    result = TechniqueResult(
        technique="HyDE", docs_retrieved=3, latency_ms=12.0, estimated_tokens=50
    )
    assert result.technique == "HyDE"
    assert result.docs_retrieved == 3
    assert result.latency_ms == 12.0
    assert result.estimated_tokens == 50


def test_rerank_returns_tuples() -> None:
    results = rerank_with_scores("rag retrieval", ["rag is about retrieval", "python"], top_k=2)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


def test_rerank_top_k_limit() -> None:
    docs = ["doc one", "doc two", "doc three", "doc four"]
    results = rerank_with_scores("query", docs, top_k=1)
    assert len(results) <= 1


def test_rerank_sorted_descending() -> None:
    docs = ["rag retrieval augmented generation", "random unrelated text", "rag is about retrieval"]
    results = rerank_with_scores("rag retrieval", docs, top_k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_higher_overlap_scores_higher() -> None:
    query = "rag retrieval"
    high_overlap = "rag retrieval augmented"
    low_overlap = "completely different topic"
    results = rerank_with_scores(query, [low_overlap, high_overlap], top_k=2)
    top_doc = results[0][0]
    assert top_doc == high_overlap


async def test_hyde_calls_llm_and_retriever() -> None:
    async def mock_llm(prompt: str) -> str:
        return "hypothetical answer"

    async def mock_retriever(query: str) -> list[str]:
        return ["doc"]

    result = await hyde_retrieval("What is RAG?", mock_llm, mock_retriever)
    assert len(result) > 0


async def test_multi_query_deduplicates() -> None:
    async def mock_llm(prompt: str) -> str:
        return "q1\nq2\nq3"

    async def mock_retriever(query: str) -> list[str]:
        return ["same doc"]

    result = await multi_query_retrieval("question", mock_llm, mock_retriever, n=3)
    assert len(result) == 1


async def test_multi_query_uses_multiple_reformulations() -> None:
    call_count = 0

    async def mock_llm(prompt: str) -> str:
        return "q1\nq2"

    async def mock_retriever(query: str) -> list[str]:
        nonlocal call_count
        call_count += 1
        return [f"doc_{call_count}"]

    await multi_query_retrieval("question", mock_llm, mock_retriever, n=2)
    assert call_count >= 2
