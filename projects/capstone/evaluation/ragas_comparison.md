# RAGAS Evaluation Report — Retrieval Strategy Comparison

## Context

50-question held-out evaluation on Kubernetes and OpenShift documentation. Questions span
single-hop factual lookups, multi-hop reasoning, named-entity specificity, and adversarial
edge cases. All strategies use the same question set and the same LLM judge (llama3.2 via
Ollama at `http://localhost:11434/v1`). Scores are means across all 50 questions; latency
is p95 measured end-to-end from query receipt to first token.

Evaluation run: 2026-07-15. Corpus: OpenShift 4.15 and Kubernetes 1.30 official docs,
~4 200 chunks after splitting at 512 tokens with 64-token overlap.

---

## Results

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Latency p95 (ms) | Cost/query (USD) | Notes |
|----------|-------------|-----------------|-------------------|-----------------|-----------------|-------|
| Naive RAG (Phase 3) | 0.71 | 0.74 | 0.68 | 210 | $0.0003 | pgvector HNSW, cosine similarity |
| + BM25 Hybrid Search | 0.74 | 0.75 | 0.73 | 240 | $0.0003 | Dense + sparse RRF fusion |
| + Cross-encoder Reranking | 0.79 | 0.76 | 0.81 | 390 | $0.0005 | BGE-reranker-large local |
| GraphRAG Global Search | 0.76 | 0.82 | 0.71 | 1250 | $0.0018 | Neo4j Leiden community reports |
| RAFT-tuned + RAG | 0.87 | 0.78 | 0.81 | 180 | $0.0002 | QLoRA Llama 3.2 3B, r=16 |

---

## Interpretation

**Production faithfulness threshold is 0.80.** Only RAFT-tuned + RAG crosses it (0.87).
All other strategies fall short and should not be used as the sole generation pipeline for
compliance-sensitive or operations-critical queries without human review.

**Hybrid + reranking is the best starting point for most teams.** Moving from naive RAG to
hybrid search with cross-encoder reranking delivers an 11% faithfulness improvement
(0.71 → 0.79) and a 19% context precision improvement (0.68 → 0.81) at roughly 2x latency
and 1.7x cost. No domain fine-tuning is required, so it can be deployed immediately.

**GraphRAG wins on answer relevancy (0.82) for thematic and multi-hop queries**, where it
surfaces community-level summaries that single-chunk retrieval misses. However, it imposes
6x latency vs naive RAG (1 250 ms p95) and is not suited for point lookups. Reserve it for
exploratory or summarisation workloads, not real-time Q&A. GraphRAG indexing costs $5–20
per 1 000 docs at commercial API rates; use local Ollama to control this cost.

**RAFT-tuned + RAG wins overall** but requires a 48–72-hour fine-tuning run upfront on
domain-specific data. The QLoRA adapter (Llama 3.2 3B, rank 16) is cheap to serve and
actually reduces latency below naive RAG (180 ms p95) because the smaller, domain-adapted
model generates accurate answers from fewer tokens. The upfront investment pays off if the
corpus is stable and query volume is high.

**Context precision is the leading indicator for reranking ROI.** The gap between naive RAG
(0.68) and + Cross-encoder Reranking (0.81) shows reranking is recovering relevant chunks
that were buried. GraphRAG's lower context precision (0.71) reflects the fact that community
reports include background context that is not directly cited in the answer.

---

## How to Regenerate

```bash
uv run python evals/ragas_harness.py
uv run python projects/phase9-domain-adaptive/capstone.py
```

The harness writes per-strategy scores to `evals/data/ragas_results.json` and prints a
summary table. Rerun after any change to the retrieval pipeline, chunking strategy, or
model.

---

## Eval Corpus Notes

- **Model**: llama3.2 (3B) via Ollama (`http://localhost:11434/v1`, `api_key="ollama"`)
- **Eval date**: 2026-07-15
- **Corpus**: OpenShift 4.15 + Kubernetes 1.30 official documentation
- **Chunk size**: 512 tokens, 64-token overlap, `RecursiveCharacterTextSplitter`
- **Embedding model**: `nomic-embed-text` via Ollama
- **Question distribution**: 40% single-hop factual, 30% multi-hop reasoning,
  20% named-entity specificity, 10% adversarial
- **RAGAS version**: 0.1.x; LLM judge and embedder both run locally — no OpenAI calls
