"""Tests for the Phase 4 capstone optimized retrieval pipeline."""

from __future__ import annotations

import asyncio

from mobility_ai.phase4.capstone import SAMPLE_CORPUS, OptimizedPipeline
from mobility_ai.phase4.cost import tier_route


def test_sample_corpus_count():
    assert len(SAMPLE_CORPUS) == 8


def test_pipeline_init():
    assert OptimizedPipeline().stats()["queries_processed"] == 0


def test_retrieve_without_hyde():
    results = asyncio.run(OptimizedPipeline().retrieve("RAG", SAMPLE_CORPUS, use_hyde=False))
    assert isinstance(results, list)


def test_retrieve_increments_counter():
    p = OptimizedPipeline()
    asyncio.run(p.retrieve("query", SAMPLE_CORPUS, use_hyde=False))
    assert p.stats()["queries_processed"] == 1


def test_cached_answer_is_not_a_retrieved_document():
    p = OptimizedPipeline()
    p.cache.set("what is HyDE", "HyDE answer")
    result = asyncio.run(p.retrieve("what is HyDE", SAMPLE_CORPUS))
    assert "HyDE answer" not in result
    assert p.cached_response("what is HyDE") == "HyDE answer"


def test_cache_response_stores():
    p = OptimizedPipeline()
    p.cache_response("q", "r")
    assert p.cache.get("q") == "r"


def test_stats_keys():
    assert {"cache", "queries_processed", "budget_remaining"} <= set(
        OptimizedPipeline().stats().keys()
    )


def test_no_match_returns_empty():
    results = asyncio.run(OptimizedPipeline().retrieve("xyzzy_nothing", SAMPLE_CORPUS))
    assert results == []


def test_tier_route_integration():
    assert tier_route("what is RAG") == "cheap"


async def test_hyde_uses_awaitable_retriever():
    async def llm(prompt):
        return "HyDE"

    pipeline = OptimizedPipeline(llm=llm)
    results = await pipeline.retrieve(
        "compare retrieval architecture", SAMPLE_CORPUS, use_hyde=True
    )
    assert results and all("HyDE" in doc for doc in results)
    assert pipeline.stats()["budget_remaining"] < 4096
