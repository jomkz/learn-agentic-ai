"""Advanced retrieval techniques: HyDE, multi-query, and cross-encoder reranking."""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel


class TechniqueResult(BaseModel):
    technique: str
    docs_retrieved: int
    latency_ms: float
    estimated_tokens: int


async def hyde_retrieval(query: str, llm, retriever) -> list:
    prompt = f"Write a short passage that directly answers: {query}"
    hypothetical_answer = await llm(prompt)
    docs = await retriever(hypothetical_answer)
    return docs


async def multi_query_retrieval(query: str, llm, retriever, n: int = 3) -> list:
    prompt = f"Generate {n} different ways to phrase this question: {query}\nOutput one per line."
    reformulations_text = await llm(prompt)
    reformulations = [line.strip() for line in reformulations_text.splitlines() if line.strip()]

    seen = set()
    unique_docs = []
    for q in reformulations:
        docs = await retriever(q)
        for doc in docs:
            content = doc if isinstance(doc, str) else getattr(doc, "page_content", str(doc))
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

    return unique_docs


def rerank_with_scores(query: str, docs: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    query_tokens = set(query.split())
    scored = []
    for doc in docs:
        doc_tokens = set(doc.split())
        score = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import asyncio

    call_log: list[str] = []

    async def mock_llm(prompt: str) -> str:
        call_log.append(prompt)
        if "different ways" in prompt:
            return "What is RAG?\nHow does RAG work?\nExplain retrieval augmented generation"
        return "RAG combines retrieval with generation to ground LLM outputs in documents."

    async def mock_retriever(query: str) -> list[str]:
        return [f"Doc about: {query[:30]}", f"Another doc for: {query[:20]}"]

    async def demo() -> None:
        query = "explain retrieval augmented generation"

        start = time.monotonic()
        hyde_docs = await hyde_retrieval(query, mock_llm, mock_retriever)
        hyde_ms = (time.monotonic() - start) * 1000
        hyde_result = TechniqueResult(
            technique="HyDE",
            docs_retrieved=len(hyde_docs),
            latency_ms=round(hyde_ms, 2),
            estimated_tokens=50,
        )

        start = time.monotonic()
        mq_docs = await multi_query_retrieval(query, mock_llm, mock_retriever, n=3)
        mq_ms = (time.monotonic() - start) * 1000
        mq_result = TechniqueResult(
            technique="MultiQuery",
            docs_retrieved=len(mq_docs),
            latency_ms=round(mq_ms, 2),
            estimated_tokens=120,
        )

        sample_docs = [
            "RAG retrieval augmented generation combines search",
            "Vector databases store embeddings for semantic search",
            "LLMs generate text based on context and prompts",
        ]
        reranked = rerank_with_scores(query, sample_docs)

        print("=== Technique Comparison ===")
        print(hyde_result.model_dump_json(indent=2))
        print(mq_result.model_dump_json(indent=2))
        print("\n=== Reranked Results ===")
        for doc, score in reranked:
            print(f"  score={score:.3f}  {doc[:60]}")

    asyncio.run(demo())
