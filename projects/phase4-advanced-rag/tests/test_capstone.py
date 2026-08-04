"""Tests for the Phase 4 capstone optimized retrieval pipeline."""

from __future__ import annotations

import asyncio

from capstone import OptimizedPipeline, SAMPLE_CORPUS
from cost import tier_route


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


def test_cache_hit_returned():
    p = OptimizedPipeline()
    p.cache.set("what is HyDE", "HyDE answer")
    result = asyncio.run(p.retrieve("what is HyDE", SAMPLE_CORPUS))
    assert result == ["HyDE answer"]


def test_cache_response_stores():
    p = OptimizedPipeline()
    p.cache_response("q", "r")
    assert p.cache.get("q") == "r"


def test_stats_keys():
    assert {"cache", "queries_processed", "budget_remaining"} <= set(OptimizedPipeline().stats().keys())


def test_no_match_returns_corpus_slice():
    results = asyncio.run(OptimizedPipeline().retrieve("xyzzy_nothing", SAMPLE_CORPUS))
    assert len(results) > 0


def test_tier_route_integration():
    assert tier_route("what is RAG") == "cheap"
