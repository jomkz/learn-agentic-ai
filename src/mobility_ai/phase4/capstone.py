"""Phase 4: async retrieval patterns, exact answer caching, and token budgeting."""

from __future__ import annotations

import asyncio

from mobility_ai.phase4.cache import CacheConfig, ExactMatchCache
from mobility_ai.phase4.cost import TokenBudget, tier_route
from mobility_ai.phase4.techniques import AsyncTextModel, rerank_with_scores

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
    def __init__(
        self, llm: AsyncTextModel | None = None, cache_config: CacheConfig | None = None
    ) -> None:
        self.llm = llm
        self.cache = ExactMatchCache(cache_config or CacheConfig())
        self.budget = TokenBudget(max_tokens=4096)
        self._queries_processed = 0

    async def retrieve(self, query: str, docs: list[str], use_hyde: bool = False) -> list[str]:
        self.budget.reset()

        async def retrieve_candidates(text: str) -> list[str]:
            return [d for d in docs if any(w in d.lower() for w in text.lower().split())]

        route = tier_route(query)
        if use_hyde and route == "expensive" and self.llm:
            from mobility_ai.phase4.techniques import hyde_retrieval

            candidates = await hyde_retrieval(
                query,
                self.llm,
                retrieve_candidates,
            )
        else:
            candidates = await retrieve_candidates(query)
        ranked = rerank_with_scores(query, candidates[:10], top_k=5)
        self._queries_processed += 1
        fitting = []
        for doc, _ in ranked:
            if self.budget.fits(doc):
                self.budget.add(doc)
                fitting.append(doc)
        return fitting

    def cached_response(self, query: str) -> str | None:
        """Answer caching is separate from retrieval so answers never become source documents."""
        return self.cache.get(query)

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
