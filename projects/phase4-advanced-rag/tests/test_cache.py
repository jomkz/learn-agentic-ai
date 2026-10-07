from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mobility_ai.phase4.cache import CacheConfig, ExactMatchCache, RedisExactMatchCache


def test_unrelated_query_regression():
    cache = ExactMatchCache()
    cache.set("What is the capital of France?", "Paris")
    assert cache.get("How do I configure TLS?") is None
    assert cache.get("What is the capital of France?") == "Paris"


def test_matching_preserves_case_and_whitespace():
    cache = ExactMatchCache()
    cache.set("US", "United States")
    assert cache.get("us") is None
    assert cache.get(" US") is None


def test_ttl_pruning_without_sleep():
    now = [0.0]
    cache = ExactMatchCache(CacheConfig(ttl_seconds=2), clock=lambda: now[0])
    cache.set("q", "r")
    now[0] = 2.0
    assert cache.get("q") is None
    assert cache.stats()["cache_size"] == 0


def test_update_does_not_duplicate_entries():
    cache = ExactMatchCache()
    cache.set("q", "old")
    cache.set("q", "new")
    assert cache.get("q") == "new"
    assert cache.stats()["cache_size"] == 1


def test_lru_eviction():
    cache = ExactMatchCache(CacheConfig(max_entries=2))
    cache.set("a", "A")
    cache.set("b", "B")
    cache.get("a")
    cache.set("c", "C")
    assert cache.get("b") is None
    assert cache.get("a") == "A"


def test_stats_and_clear():
    cache = ExactMatchCache()
    cache.set("q", "r")
    cache.get("q")
    cache.get("missing")
    assert cache.stats()["hit_rate"] == 0.5
    cache.clear()
    assert cache.stats()["cache_size"] == cache.stats()["hits"] == 0


@pytest.mark.parametrize("kwargs", [{"max_entries": 0}, {"ttl_seconds": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        CacheConfig(**kwargs)


def test_redis_backing_store_is_shared_and_namespaced():
    storage = {}
    client = MagicMock()
    client.set.side_effect = lambda key, value, **kwargs: storage.update({key: value})
    client.get.side_effect = storage.get
    writer = RedisExactMatchCache(client, namespace="model-v1:corpus-v1:tenant-a")
    reader = RedisExactMatchCache(client, namespace="model-v1:corpus-v1:tenant-a")
    other = RedisExactMatchCache(client, namespace="model-v1:corpus-v2:tenant-a")
    writer.set("q", "answer")
    assert reader.get("q") == "answer"
    assert other.get("q") is None
    assert client.set.call_args.kwargs["ex"] == 3600


def test_redis_errors_are_not_hidden():
    client = MagicMock()
    client.get.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        RedisExactMatchCache(client, namespace="test").get("q")
