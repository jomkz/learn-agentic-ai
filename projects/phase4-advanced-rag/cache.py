"""Semantic caching for LLM responses. Reduces cost and latency on repeated queries."""

from __future__ import annotations

import hashlib
import time

from pydantic import BaseModel


class CacheConfig(BaseModel):
    similarity_threshold: float = 0.85
    ttl_seconds: int = 3600
    max_entries: int = 1000


class CacheEntry(BaseModel):
    query: str
    response: str
    embedding: list[float]
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._store: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0

    def _embed(self, text: str) -> list[float]:
        # Deterministic hash-based pseudo-embedding — replace with real embeddings in production
        raw = hashlib.md5(text.lower().strip().encode()).digest()
        return [b / 255.0 for b in raw]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query: str) -> str | None:
        emb = self._embed(query)
        now = time.time()
        best_score = 0.0
        best_entry: CacheEntry | None = None
        for entry in self._store:
            if now - entry.timestamp > self.config.ttl_seconds:
                continue
            score = self._cosine_similarity(emb, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score >= self.config.similarity_threshold:
            best_entry.hit_count += 1
            self._hits += 1
            return best_entry.response
        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        emb = self._embed(query)
        self._store.append(
            CacheEntry(query=query, response=response, embedding=emb, timestamp=time.time())
        )
        if len(self._store) > self.config.max_entries:
            self._store.pop(0)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cache_size": len(self._store),
        }

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


def build_redis_cache(redis_url: str = "redis://localhost:6379") -> SemanticCache | None:
    """Connect to Redis; return in-memory SemanticCache if Redis is unavailable."""
    try:
        import redis

        r = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
        r.ping()
        return SemanticCache()
    except Exception:
        return None
