"""Exact-match response caching. No semantic similarity is inferred from hashes."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    ttl_seconds: int = Field(default=3600, gt=0)
    max_entries: int = Field(default=1000, gt=0)
    # Include model, prompt, corpus version and tenant when sharing cached answers.
    namespace: str = Field(default="demo-v1", min_length=1)


class ExactMatchCache:
    def __init__(self, config: CacheConfig | None = None, *, clock: Callable = time.monotonic):
        self.config = config or CacheConfig()
        self._clock = clock
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _prune(self) -> None:
        now = self._clock()
        for key in [key for key, (_, expiry) in self._store.items() if expiry <= now]:
            del self._store[key]

    def get(self, query: str) -> str | None:
        self._prune()
        if query in self._store:
            self._hits += 1
            self._store.move_to_end(query)
            return self._store[query][0]
        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        self._prune()
        self._store[query] = (response, self._clock() + self.config.ttl_seconds)
        self._store.move_to_end(query)
        while len(self._store) > self.config.max_entries:
            self._store.popitem(last=False)

    def stats(self) -> dict:
        self._prune()
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "cache_size": len(self._store),
            "backend": "memory-exact",
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = 0


class RedisExactMatchCache:
    """Shared Redis entries with server-enforced TTL; errors propagate to the caller.

    Redis capacity/eviction is configured on the server (maxmemory/maxmemory-policy).
    This class does not pretend the per-process memory cache's capacity applies to Redis.
    """

    def __init__(self, client: Any, *, namespace: str, ttl_seconds: int = 3600):
        if not namespace or ttl_seconds <= 0:
            raise ValueError("namespace and positive ttl_seconds are required")
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.prefix = (
            "mobility-ai:responses:" + hashlib.sha256(namespace.encode()).hexdigest() + ":"
        )

    def _key(self, query: str) -> str:
        return self.prefix + hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        value = self.client.get(self._key(query))
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def set(self, query: str, response: str) -> None:
        self.client.set(self._key(query), response, ex=self.ttl_seconds)

    def clear(self) -> None:
        for key in self.client.scan_iter(match=self.prefix + "*", count=100):
            self.client.delete(key)


def build_redis_cache(
    redis_url: str = "redis://localhost:6379",
    *,
    namespace: str = "demo-v1",
    ttl_seconds: int = 3600,
) -> RedisExactMatchCache:
    """Create a real Redis backend. Unavailable Redis raises, never silently falls back."""
    import redis

    client = redis.Redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
    client.ping()
    return RedisExactMatchCache(client, namespace=namespace, ttl_seconds=ttl_seconds)
