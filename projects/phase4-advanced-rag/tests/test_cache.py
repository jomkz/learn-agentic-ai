from __future__ import annotations

import time

import pytest
from cache import CacheConfig, SemanticCache, build_redis_cache


def test_cache_config_defaults():
    assert CacheConfig().similarity_threshold == 0.85


def test_cache_initially_empty():
    assert SemanticCache().stats()["cache_size"] == 0


def test_set_increases_size():
    cache = SemanticCache()
    cache.set("q", "r")
    assert cache.stats()["cache_size"] == 1


def test_get_miss_returns_none():
    cache = SemanticCache()
    assert cache.get("anything") is None


def test_get_miss_increments_misses():
    cache = SemanticCache()
    cache.get("anything")
    assert cache.stats()["misses"] == 1


def test_exact_query_hit():
    cache = SemanticCache()
    cache.set("hello", "world")
    assert cache.get("hello") == "world"


def test_hit_increments_hits():
    cache = SemanticCache()
    cache.set("hello", "world")
    cache.get("hello")
    assert cache.stats()["hits"] == 1


def test_hit_rate_after_one_hit_one_miss():
    cache = SemanticCache()
    cache.set("hello", "world")
    cache.get("hello")  # hit
    cache.get("zzzzz")  # miss (dissimilar enough to not match)
    assert cache.stats()["hit_rate"] == pytest.approx(0.5)


def test_clear_resets_all():
    cache = SemanticCache()
    cache.set("hello", "world")
    cache.get("hello")
    cache.clear()
    s = cache.stats()
    assert s["cache_size"] == 0
    assert s["hits"] == 0
    assert s["misses"] == 0


def test_embed_is_deterministic():
    cache = SemanticCache()
    assert cache._embed("test") == cache._embed("test")


def test_embed_different_texts_differ():
    cache = SemanticCache()
    assert cache._embed("hello") != cache._embed("world")


def test_cosine_similarity_same_vector():
    cache = SemanticCache()
    assert cache._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    cache = SemanticCache()
    assert cache._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    cache = SemanticCache()
    assert cache._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_max_entries_evicts_oldest():
    config = CacheConfig(max_entries=2)
    cache = SemanticCache(config=config)
    cache.set("first", "a")
    cache.set("second", "b")
    cache.set("third", "c")
    assert cache.stats()["cache_size"] == 2


def test_build_redis_cache_returns_cache_or_none():
    result = build_redis_cache("redis://localhost:6379")
    assert result is None or isinstance(result, SemanticCache)


def test_ttl_expiry():
    config = CacheConfig(ttl_seconds=0)
    cache = SemanticCache(config=config)
    cache.set("expiring", "value")
    time.sleep(0.01)
    assert cache.get("expiring") is None


def test_threshold_blocks_dissimilar():
    config = CacheConfig(similarity_threshold=0.99)
    cache = SemanticCache(config=config)
    cache.set("apple pie", "a dessert")
    assert cache.get("quantum physics") is None
