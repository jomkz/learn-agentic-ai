# ADR-001: Vector Store Selection

**Status**: Accepted
**Date**: 2026-01-15
**Deciders**: Platform team

---

## Context

The platform requires a production-grade vector store to index dense embeddings for semantic
retrieval over an enterprise document corpus. The corpus is expected to grow to 10–50 million
chunks. Requirements are:

- Sub-20ms ANN query latency at the p95 for top-20 retrieval
- Metadata filtering (date range, document type, source, access-control label)
- Horizontal scalability or vertical scalability path to 50M+ vectors
- Operational simplicity: preference for components already present in the stack
- Hybrid retrieval support (dense + sparse in same query path, or easy composition)
- Strong Kubernetes/OpenShift operator support for production deployment

Candidate options evaluated:

| Option | License | Managed on OpenShift | Notes |
|--------|---------|----------------------|-------|
| pgvector (PostgreSQL extension) | MIT | Yes (StatefulSet) | Extends existing PostgreSQL |
| Qdrant | Apache 2.0 | Yes (Helm chart) | Purpose-built; Rust; payload filtering |
| Milvus | Apache 2.0 | Yes (Milvus Operator) | Horizontally scalable; operationally heavy |
| Weaviate | BSD | Yes (Helm chart) | GraphQL API; built-in modules |
| Chroma | Apache 2.0 | Limited | Designed for local/dev; not production-hardened |

---

## Decision

**Primary store: pgvector.** Use Qdrant as a secondary store for high-cardinality filtered queries.

### pgvector as primary

pgvector's HNSW index (available since PostgreSQL 16) provides ANN performance comparable to
purpose-built vector databases at corpus sizes up to approximately 20 million vectors. At the
expected initial corpus size (< 5M chunks) pgvector comfortably meets the latency target.

The decisive factor is operational co-location: the platform already runs PostgreSQL for the
LangGraph checkpoint store, MLflow backend, and application metadata. Adding pgvector as a
PostgreSQL extension requires no new infrastructure, no new operator, and no new backup strategy.
A single SQL query can join vector similarity with structured metadata filters, which is the
dominant retrieval pattern for this workload.

HNSW parameters: `m=16`, `ef_construction=64`. Query parameter `hnsw.ef_search=40` gives a good
recall/latency trade-off for top-20 retrieval. These values should be re-evaluated if the corpus
grows beyond 10M chunks.

### Qdrant as secondary

pgvector's filtered HNSW can degrade significantly when filters eliminate a large fraction of
the index (the "filtering cliff"). Qdrant's payload-indexed filtered ANN does not have this
property: it maintains consistent performance regardless of how selective the filter is.

The retrieval layer includes a query router that directs requests with more than two metadata
filter predicates to Qdrant. Both stores are seeded from the same KFP ingestion pipeline;
Qdrant receives the same vectors and payload metadata as pgvector.

### Options not selected

**Milvus**: operationally heavier (requires etcd, MinIO for internal storage, separate components
for each role). The performance advantage over pgvector at our corpus size does not justify the
operational overhead.

**Weaviate**: the GraphQL-first API adds friction for a Python-first team. The module ecosystem
is useful but not differentiated enough to outweigh the API ergonomics cost.

**Chroma**: not production-hardened. Lacks multi-node support. Suitable for notebooks and local
development (used in Phase 3) but not for this deployment.

---

## Consequences

### What becomes easier

- A single PostgreSQL connection pool services both metadata queries and vector queries, reducing
  connection overhead and simplifying the application data layer.
- Backup and restore of the vector index is handled by the existing PostgreSQL backup job (pg_dump
  or continuous WAL archiving), with no separate process for the vector store.
- SQL-native metadata filtering is transparent to developers already familiar with SQLAlchemy.
- pgvector is supported by LangChain's `PGVector` integration out of the box.

### What becomes harder

- If the corpus grows beyond ~20M vectors, pgvector HNSW recall or latency may degrade and
  migration to a purpose-built store becomes necessary. This migration should be planned at the
  15M vector mark.
- Running two vector stores (pgvector + Qdrant) doubles the operational surface for vector
  data. Teams must ensure both stores remain in sync after every ingestion run.
- Qdrant's Rust-based client API differs from the SQLAlchemy-based pgvector integration. The
  retrieval abstraction layer must handle both backends transparently.
- HNSW index rebuilds on pgvector are blocking operations. Large corpus additions during peak
  query hours should be scheduled during off-hours maintenance windows.
