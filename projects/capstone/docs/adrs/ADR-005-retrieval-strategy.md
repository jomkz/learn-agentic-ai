# ADR-005 — Retrieval Strategy

**Status:** Accepted
**Date:** 2026-07-15
**Deciders:** Platform AI team

## Context

The production RAG pipeline must serve a mixed query workload: precise factual lookups (keyword-heavy),
semantic similarity queries (concept-heavy), and thematic synthesis queries ("what are all the known failure
modes of X?"). A single retrieval strategy does not perform optimally across all three. In addition, cost
and latency budgets limit how many retrieval stages can be applied per request.

Requirements:
- P95 retrieval latency: under 300ms excluding LLM generation.
- Faithfulness (RAGAS): target ≥ 0.85 on the held-out evaluation set.
- Support corpus updates (new documents) without pipeline downtime.
- Reduce per-query LLM token cost at scale via caching.

## Options Evaluated

### Option A — Dense only (pgvector cosine)

Embed queries with `text-embedding-3-small`; retrieve top-k chunks via pgvector cosine similarity.

**Pros:** Simple; fast (single ANN query); easy to update (embed and upsert new chunks).
**Cons:** Misses exact keyword matches ("error code OCP-4531"); poor recall on rare terms; no re-ranking.

### Option B — Dense + BM25 hybrid (Reciprocal Rank Fusion)

Run dense (pgvector) and sparse (BM25 via OpenSearch or pgvector `sparsevec`) retrieval in parallel;
merge results with Reciprocal Rank Fusion (RRF).

**Pros:** Best-of-both-worlds recall; RRF is parameter-free and robust; handles both semantic and keyword
queries well; well-supported in LangChain and LlamaIndex.
**Cons:** Requires a BM25 index in addition to the vector index (managed in the same PostgreSQL instance
via `pg_bm25` / ParadeDB, or a separate OpenSearch instance); adds ~20–40ms per query.

### Option C — Dense + cross-encoder reranking

Retrieve a larger candidate set (top-20 or top-50) with dense retrieval, then rerank with a cross-encoder
(e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or Cohere Rerank API).

**Pros:** Significant precision improvement over dense-only; cross-encoders capture query-document
interaction directly.
**Cons:** Cross-encoder inference scales linearly with candidate set size; adds 50–100ms for a 20-doc
candidate set; no keyword recall improvement.

### Option D — GraphRAG local search

For a given entity mentioned in the query, retrieve its neighborhood in the Neo4j entity graph plus
associated community reports.

**Pros:** Excellent for entity-centric queries; surfaces relationships dense retrieval misses.
**Cons:** Requires a pre-built entity graph; entity linking from free-text query is error-prone; high
indexing cost; overkill for most factual lookups.

### Option E — GraphRAG global search

Generate a thematic answer by aggregating all community reports in the Neo4j graph.

**Pros:** Only option that can synthesize across the entire corpus; answers "what are the main themes?" type
queries that have no single-chunk answer.
**Cons:** Very high token cost (dozens of community reports sent to the LLM); slow (3–10 seconds); not
suitable as a default path.

## Decision

**Default pipeline: Option B (hybrid dense + BM25) with Option C (cross-encoder reranking) on top. Options
D and E are activated selectively by a query classifier.**

### Standard pipeline (all queries)

1. Embed query with `text-embedding-3-small`.
2. Run dense ANN (pgvector) and BM25 (ParadeDB `pg_bm25`) retrieval in parallel; top-20 candidates each.
3. Merge with RRF to produce a unified top-30 candidate list.
4. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` (self-hosted, batched); select top-5 chunks.
5. Check Redis semantic cache: if a semantically equivalent query (cosine > 0.97) was answered recently,
   return the cached response and skip LLM generation.

### Thematic query path (classifier-activated)

A lightweight binary classifier (fine-tuned `deberta-v3-small`) routes queries flagged as thematic to
GraphRAG global search (Option E). Classifier adds ~10ms; GraphRAG global search replaces step 4–5 above.

### Entity query path (classifier-activated)

Queries mentioning specific named entities (operators, error codes, cluster names) are augmented with
GraphRAG local search (Option D) results appended to the standard reranked chunks.

## Consequences

**What becomes easier:**
- Hybrid retrieval handles both keyword-heavy (BM25 wins) and concept-heavy (dense wins) queries without
  tuning per-query-type retrieval parameters.
- Cross-encoder reranking raises faithfulness on the evaluation set by approximately 8–12 percentage
  points vs. dense-only, based on Phase 3 ablation results.
- Redis semantic cache reduces average per-query LLM token cost by 30–40% at scale (empirical estimate
  from similar production deployments).
- GraphRAG global search is available for the class of queries that cannot be answered by any chunk
  retrieval strategy, without adding cost to the standard path.

**What becomes harder:**
- Hybrid + reranking adds approximately 100ms per query vs. dense-only (target p95: 250ms for the full
  retrieval stack).
- GraphRAG indexing is expensive ($5–20 per 1 000 documents at API rates) and must be re-run when the
  corpus changes significantly; incremental re-indexing is not yet well-supported by the Microsoft
  GraphRAG library.
- The semantic cache requires a separate Redis instance and a cache key embedding store; cache
  invalidation policy (TTL vs. content-hash) must be chosen carefully to avoid serving stale answers
  after corpus updates.
- Three retrieval backends (pgvector, ParadeDB BM25, Neo4j) each require monitoring, schema migrations,
  and backup procedures.
- The query classifier adds a dependency: if it misclassifies, thematic queries fall back to standard
  pipeline (degraded quality) or standard queries hit the expensive GraphRAG path (cost overrun).
