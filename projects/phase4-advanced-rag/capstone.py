"""Phase 4 capstone: Optimized retrieval combining HyDE, reranking, semantic cache, and cost tiering."""

from __future__ import annotations

import asyncio

from cache import CacheConfig, SemanticCache
from cost import TokenBudget, estimate_cost_usd, tier_route
from techniques import rerank_with_scores

SAMPLE_CORPUS: list[str] = [
    "RAG retrieves documents to ground LLM outputs in factual sources.",
    "HyDE generates a hypothetical answer and uses its embedding for retrieval.",
    "Multi-query retrieval generates reformulations and fuses results with RRF.",
    "Cross-encoder reranking scores query-document pairs for precision at top-k.",
    "Semantic caching returns cached responses for similar queries to reduce cost.",
    "Token budgeting prevents context window overflow in long RAG pipelines.",
    "Model tiering routes simple queries to cheap models and complex ones to expensive models.",
    "DSPy optimizes prompt programs using labeled examples and a faithfulness metric.",
]


class OptimizedPipeline:
    def __init__(self, llm=None, cache_config: CacheConfig | None = None) -> None:
        self.llm = llm
        self.cache = SemanticCache(cache_config or CacheConfig())
        self.budget = TokenBudget(max_tokens=4096)
        self._queries_processed = 0

    async def retrieve(self, query: str, docs: list[str], use_hyde: bool = False) -> list[str]:
        cached = self.cache.get(query)
        if cached:
            return [cached]
        route = tier_route(query)
        if use_hyde and route == "expensive" and self.llm:
            from techniques import hyde_retrieval

            candidates = await hyde_retrieval(
                query,
                self.llm,
                lambda q: [d for d in docs if any(w in d.lower() for w in q.lower().split())],
            )
        else:
            candidates = [
                d for d in docs if any(w in d.lower() for w in query.lower().split())
            ]
            if not candidates:
                candidates = docs[:3]
        ranked = rerank_with_scores(query, candidates[:10], top_k=5)
        self._queries_processed += 1
        return [doc for doc, _ in ranked]

    def cache_response(self, query: str, response: str) -> None:
        self.cache.set(query, response)

    def stats(self) -> dict:
        return {
            "cache": self.cache.stats(),
            "queries_processed": self._queries_processed,
            "budget_remaining": self.budget.remaining(),
        }


if __name__ == "__main__":
    p = OptimizedPipeline()
    results = asyncio.run(p.retrieve("what is HyDE", SAMPLE_CORPUS))
    print(f"Retrieved {len(results)} docs")
    print(p.stats())
