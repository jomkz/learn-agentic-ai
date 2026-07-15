# Enterprise Knowledge Assistant Platform — System Architecture

## Overview

The Enterprise Knowledge Assistant Platform (EKAP) is a production-grade, multi-agent RAG system
designed to answer complex questions over a heterogeneous corpus of enterprise documents. It
integrates every major component developed across the learning path: document ingestion pipelines,
hybrid retrieval, domain-adapted LLMs, a LangGraph multi-agent supervisor, MCP-based tooling,
safety layers, and full-stack observability.

The platform is deployed on OpenShift AI and is built around open-source components throughout,
with Podman/Kubernetes as the container runtime and vLLM as the primary inference backend.

---

## Component Overview

### 1. Document Ingestion and Parsing

Raw documents (PDF, DOCX, HTML, Markdown, PPTX) are ingested from an S3-compatible object store
(MinIO in local dev; Red Hat OpenShift Data Foundation in production). Docling handles parsing and
structured extraction, preserving document hierarchy, tables, and figures as structured JSON.

Each parsed document is chunked with a recursive character splitter configured for 512-token
chunks with a 64-token overlap. Metadata extracted at parse time — document type, source, date,
section title, confidence score — is preserved through the entire pipeline and stored alongside
vectors for filtered retrieval.

### 2. Data Pipeline (Kubeflow Pipelines on OpenShift AI)

The ingestion pipeline is orchestrated by Kubeflow Pipelines (KFP) running on OpenShift AI. The
pipeline DAG consists of five sequential stages:

| Stage | Component | Description |
|-------|-----------|-------------|
| fetch | MinIO SDK | Pull raw files from the corpus bucket |
| parse | Docling | Extract structured text, tables, figures |
| chunk | LangChain splitter | Recursive character splitting with metadata |
| embed | Ray Data + BGE-M3 | Distributed embedding over worker pool |
| store | pgvector + Qdrant | Write dense and sparse vectors to stores |

The pipeline is triggered by S3 event notifications. New documents in the corpus bucket fire a KFP
pipeline run automatically. Pipeline runs are tracked in MLflow for reproducibility.

### 3. Embedding (Ray Data Distributed)

Embedding is parallelised across a RayCluster (see `deploy/raycluster.yaml`) with three GPU worker
nodes. Each worker runs BGE-M3, which produces both dense and sparse representations in a single
forward pass, enabling native hybrid retrieval without a separate sparse encoder.

The Ray actor pool batches documents in groups of 32 and writes results directly to the vector
stores via async writes. Total throughput on the three-GPU cluster is approximately 2,000
documents/minute for standard enterprise PDFs.

### 4. Vector Storage

Two vector stores serve complementary roles:

**pgvector (primary)** stores dense embeddings co-located with document metadata in PostgreSQL.
The HNSW index (m=16, ef_construction=64) gives sub-10ms ANN queries at the document corpus
sizes expected in this deployment. Storing vectors in PostgreSQL means a single SQL join can
combine vector similarity with structured metadata filters, which is critical for date-range and
source-type filtering. See ADR-001 for the full decision rationale.

**Qdrant (secondary)** handles high-cardinality filtered queries where pgvector's filtered HNSW
performance degrades. Qdrant's payload-indexed filtering does not suffer the same performance
cliff. The retrieval layer routes to Qdrant automatically when the query carries more than two
metadata filter predicates.

Both stores are seeded from the same KFP pipeline run and kept in sync on every ingestion.

### 5. Hybrid Retrieval and Reranking

At query time the retrieval layer issues parallel queries to both stores:

- **Dense retrieval**: top-20 candidates from pgvector HNSW (cosine similarity)
- **Sparse retrieval**: top-20 candidates from a BM25 index maintained in Elasticsearch
- **Fusion**: Reciprocal Rank Fusion (RRF) merges the two ranked lists into a combined top-40

The fused candidate set is then passed to a BGE reranker (cross-encoder) that scores each
candidate against the original query. The top-5 reranked chunks are passed to the LLM for
synthesis. This pipeline consistently outperforms dense-only retrieval on faithfulness metrics
(see `evaluation/ragas_comparison.md`).

### 6. Semantic Cache (Redis)

A Redis-backed semantic cache sits in front of the retrieval pipeline. Incoming query embeddings
are compared against cached query embeddings using cosine similarity (threshold 0.92). A cache
hit returns the cached retrieval result and LLM response directly, bypassing the full pipeline.

This reduces median latency from ~2.5s to ~80ms for repeated or near-duplicate queries, and
cuts LLM API cost substantially in deployments with high query overlap (common in enterprise
helpdesk scenarios).

### 7. Model Serving (vLLM on OpenShift AI)

The primary inference backend is vLLM deployed as a Kubernetes Deployment (see
`deploy/vllm-deployment.yaml`). vLLM's PagedAttention memory manager enables high-throughput
continuous batching, which is essential for the multi-agent workload where several parallel
LLM calls occur per user query.

The base model is `meta-llama/Llama-3.1-8B-Instruct`. A LoRA adapter (`raft-adapter`) trained
with RAFT fine-tuning (Phase 9) is loaded at startup via `--lora-modules`. The adapter is hot-
swappable: updating the adapter path and redeploying the Deployment (or using vLLM's model swap
API) requires no downtime.

A second, smaller model (`Llama-3.2-1B-Instruct`) runs on a separate vLLM instance for
classification tasks (intent routing, safety pre-screen) where latency matters more than
generation quality.

### 8. Agent Orchestration (LangGraph Multi-Agent Supervisor)

The agent layer is implemented as a LangGraph StateGraph with a supervisor pattern. See ADR-003
for the orchestration framework decision. The graph consists of:

- **Supervisor node**: routes the incoming query to one or more specialist agents based on
  intent classification. Returns a final response by merging agent outputs.
- **Research agent**: performs cache lookup, retrieval, reranking, and LLM synthesis. Emits
  cited answers with source provenance.
- **Writing agent**: takes a research agent output and reformats it for the requested output
  mode (executive summary, detailed explanation, bullet list, JSON schema).
- **Calculation agent**: handles numeric reasoning tasks by invoking a Python REPL tool.
- **Verification agent**: cross-checks claims in the research agent output against a second
  retrieval pass with a different query reformulation.

LangGraph checkpointing (backed by PostgreSQL) persists the full graph state across turns,
enabling multi-turn conversations and human-in-the-loop review at any edge. Conditional edges
implement retry logic and fallback routing when an agent returns a low-confidence result.

### 9. Tool Layer (MCP Server)

Agents access external capabilities through an MCP server that exposes the following tools:

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Hybrid retrieval + reranking pipeline |
| `get_document` | Fetch full document by ID from the object store |
| `calculate` | Python REPL for numeric reasoning |
| `search_web` | Tavily-backed web search for current information |
| `query_database` | Read-only SQL against the enterprise data warehouse |
| `get_user_context` | Fetch caller's org unit, role, document permissions |

The MCP server runs as a sidecar alongside the LangGraph application and is registered at
startup. Tool call traces are forwarded to LangSmith.

### 10. Safety Layer (LlamaGuard + NeMo Guardrails)

Safety operates at two levels:

**Input safety (LlamaGuard)**: every incoming user query is classified by LlamaGuard before it
reaches the supervisor. Queries classified as unsafe (prompt injection, PII exfiltration attempts,
policy violations) are rejected with a templated response. LlamaGuard runs on the small vLLM
instance to keep p99 latency impact under 100ms.

**Output guardrails (NeMo Guardrails)**: the Colang-defined rail set enforces factual grounding
(no hallucinated citations), topic restriction (stay within document corpus), and tone policies.
Rails are evaluated on every LLM output before it is streamed to the caller.

### 11. Observability

**LangSmith**: all LangGraph traces — node inputs/outputs, tool calls, latencies, token counts —
are forwarded to LangSmith. Trace-level RAGAS faithfulness scores are computed inline using the
`evals/ragas_harness.py` evaluator and attached as metadata.

**OpenTelemetry**: the application emits OTLP traces and metrics to an OpenTelemetry Collector
running in the cluster. Traces flow to Jaeger; metrics flow to Prometheus.

**Prometheus + Grafana**: the vLLM `/metrics` endpoint is scraped by Prometheus. The Grafana
dashboard covers: `vllm:request_throughput`, `vllm:e2e_request_latency_seconds_bucket` (p50/p95/
p99), GPU utilisation, cache hit rate, and per-agent call counts. Alert rules fire on p95 latency
above 5s and error rate above 1%.

**MLflow Model Registry**: every trained LoRA adapter and every RAGAS evaluation run is logged to
MLflow. The vLLM deployment pulls the adapter tagged `production` from the registry. Promoting a
new adapter to `production` triggers a rolling restart of the vLLM Deployment.

---

## Data Flow

```
User Query
    |
    v
[LangGraph Supervisor]
    |
    +--> [LlamaGuard safety check]
    |         |
    |         +-- UNSAFE --> reject (templated response)
    |         |
    |         +-- SAFE --> continue
    |
    +--> [Research Agent]
    |         |
    |         +--> [Semantic Cache (Redis)] -- HIT --> return cached result
    |         |         |
    |         |         +-- MISS --> continue
    |         |
    |         +--> [Hybrid Retrieval]
    |         |         |- Dense: pgvector HNSW top-20
    |         |         |- Sparse: BM25 top-20
    |         |         `- Fusion: RRF -> top-40
    |         |
    |         +--> [BGE Cross-Encoder Reranker] -> top-5 chunks
    |         |
    |         +--> [vLLM Synthesis] -> cited answer
    |         |
    |         +--> [NeMo Guardrails] -- FAIL --> retry or reject
    |         |
    |         +--> [Cache write (Redis)]
    |
    +--> [Writing Agent] -> formatted response
    |
    v
[SSE stream to caller]
    |
    v
[LangSmith trace + OTel metrics]
```

---

## Deployment Topology

| Component | Where | Replicas |
|-----------|-------|----------|
| LangGraph app | OpenShift Deployment | 2 |
| vLLM (8B) | OpenShift Deployment, GPU node | 1 |
| vLLM (1B) | OpenShift Deployment, GPU node | 1 |
| MCP server | Sidecar in LangGraph pod | 2 |
| RayCluster (embedding) | KubeRay on OpenShift | 1 head + 3 workers |
| PostgreSQL + pgvector | OpenShift StatefulSet | 1 (+ replica) |
| Qdrant | OpenShift StatefulSet | 3 (distributed) |
| Redis | OpenShift StatefulSet | 1 (Sentinel in prod) |
| Neo4j | OpenShift StatefulSet | 1 |
| MLflow | OpenShift Deployment | 1 |
| OTel Collector | DaemonSet | 1 per node |
| Grafana + Prometheus | OpenShift monitoring stack | managed |

---

## Security Posture

- All inter-service communication uses mTLS via OpenShift Service Mesh (Istio).
- The vLLM endpoint is not exposed externally; only the LangGraph app has network access to it.
- Document access control is enforced at retrieval time via the `get_user_context` MCP tool;
  chunks from documents the caller cannot access are filtered before reranking.
- API keys and the HuggingFace Hub token are stored in OpenShift Secrets, never in environment
  variables baked into images.
- Garak vulnerability scans run nightly against the vLLM endpoint to detect prompt injection,
  jailbreak, and data leakage regressions.
