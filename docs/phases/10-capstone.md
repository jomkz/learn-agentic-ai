# Final Capstone: Enterprise Knowledge Assistant Platform

**Duration: 2 weeks** | [← Phase 9](09-graphrag-raft-instructlab.md) | [← Index](../index.md)

**Project directory:** [`projects/`](../../projects/) — draws from all phase project directories

---

## Objective

Combine every phase into a single production-architected system. This is not a new build from scratch — it is the integration, hardening, and documentation of the components already built across Phases 1-9.

---

## System Architecture

### Component Map

| Concern | Technology | Phase Built |
|---------|-----------|------------|
| Document parsing | Docling | Phase 3 |
| Data pipeline | KubeFlow Pipelines on OpenShift AI | Phase 8 |
| Distributed embedding | Ray Data | Phase 8 |
| Vector storage | pgvector (dense) + Qdrant (filtered) | Phase 3 |
| Hybrid retrieval | pgvector dense + BM25 sparse + GraphRAG global | Phases 3, 9 |
| Reranking | Cross-encoder (BGE or Cohere) | Phase 4 |
| Semantic cache | Redis Semantic Cache | Phase 4 |
| Model serving | vLLM on OpenShift AI | Phase 8 |
| Domain adaptation | RAFT or InstructLab fine-tuned model | Phase 9 |
| Provider abstraction | LlamaStack distribution → vLLM | Phases 7, 8 |
| Agent orchestration | LangGraph multi-agent | Phase 5 |
| Tool layer | Custom MCP server | Phase 6 |
| Safety | LlamaGuard + NeMo Guardrails | Phases 6, 7 |
| Context management | Token budget manager | Phase 6 |
| Tracing | LangSmith | Phase 2 |
| Model registry | MLflow | Phase 8 |
| Metrics | Prometheus + Grafana | Phase 8 |
| Evaluation | RAGAS harness | Phase 3+ |
| Containerization | Podman | Phase 8 |

### Data Flow

```
User Query
    ↓
LangGraph Supervisor Agent
    ├── Safety check (NeMo Guardrails)
    ├── Context budget check (token manager)
    │
    ├── Research Agent
    │       ├── Semantic cache check (Redis) → cache hit → return
    │       ├── Query classification → routing strategy
    │       ├── Dense retrieval (pgvector)
    │       ├── Sparse retrieval (BM25)
    │       ├── GraphRAG global search (Neo4j) [for thematic queries]
    │       ├── Cross-encoder reranking
    │       └── LLM synthesis (vLLM via LlamaStack)
    │
    ├── Analysis Agent (code interpreter)
    │       └── Executes Python for data analysis, chart generation
    │
    └── Writing Agent
            └── Structures and formats the final response
    ↓
Output guardrails (Guardrails AI)
    ↓
SSE stream to client
```

### Ingestion Pipeline (KubeFlow Pipelines on OpenShift AI)
```
New documents (S3/MinIO)
    ↓ KFP trigger (schedule or webhook)
Stage 1: Fetch & Docling parsing → structured JSON
Stage 2: Chunking + metadata enrichment (Ray Data, distributed)
Stage 3: Embedding generation (Ray Serve, autoscaled)
Stage 4: Write to pgvector + Qdrant
Stage 5: GraphRAG entity extraction + Neo4j update [async, expensive]
    ↓
MLflow: log pipeline run, data version, chunk counts
```

---

## Deliverables

### 1. Architecture Documentation (`projects/capstone/docs/`)

- `architecture.md`: written description of component choices and their justification
- `architecture-component-view.png`: diagram showing components and their relationships
- `architecture-dataflow.png`: diagram showing the ingestion and query data flows

### 2. Architectural Decision Records (`projects/capstone/docs/adrs/`)

One ADR per major architectural choice, using this template:

```markdown
# ADR-001: Vector Store Selection

## Status: Accepted

## Context
[What problem were we solving? What options existed?]

## Decision
[What did we choose and why?]

## Consequences
[What becomes easier? What becomes harder?]
```

Suggested ADRs:
- ADR-001: Vector store selection (pgvector vs. Qdrant vs. Milvus)
- ADR-002: Domain adaptation approach (RAG vs. RAFT vs. InstructLab)
- ADR-003: Agent orchestration (LangGraph vs. LlamaStack native agents)
- ADR-004: Guardrails strategy (NeMo vs. Guardrails AI vs. LlamaGuard)
- ADR-005: Retrieval strategy (dense vs. hybrid vs. GraphRAG)

### 3. Deployment Manifests (`projects/capstone/deploy/`)

OpenShift AI Kubernetes manifests:
- `vllm-deployment.yaml`: vLLM `Deployment`, `Service`, `Route`
- `llamastack-deployment.yaml`: LlamaStack server `Deployment`, `Service`
- `redis-deployment.yaml`: Redis `StatefulSet` + `Service`
- `neo4j-deployment.yaml`: Neo4j `StatefulSet` + `Service`
- `kfp-pipeline.yaml`: compiled KFP pipeline YAML
- `raycluster.yaml`: `RayCluster` CRD for distributed embedding

Local dev alternative: the root `podman-compose.yml` covers all services.

### 4. Evaluation Report (`projects/capstone/evaluation/`)

- `ragas_results.json`: RAGAS scores for each retrieval strategy on a 50-question held-out set
- `ragas_comparison.md`: written interpretation — which strategy wins and under what conditions
- `latency_benchmarks.md`: p50/p95/p99 latency per retrieval strategy at 10/50/100 concurrent queries
- `cost_analysis.md`: estimated token cost per query for each strategy (important for production sizing)

### 5. Runbook (`projects/capstone/RUNBOOK.md`)

Operational procedures:
- How to start local dev environment (`podman-compose up`)
- How to run the KFP ingestion pipeline
- How to update the domain-adapted model (InstructLab → vLLM hot-swap procedure)
- How to add a new document corpus to the knowledge base
- How to roll back a bad model version in MLflow
- How to read the Grafana dashboard and respond to alerts

---

## Suggested Week 1 / Week 2 Split

**Week 1 — Integration**
- Wire all components together; fix integration issues that arise at the seams
- Run the full query flow end-to-end at least once
- Run the ingestion KFP pipeline with a real document set
- Verify LangSmith traces show the complete multi-agent execution

**Week 2 — Hardening and Documentation**
- Run the full RAGAS evaluation suite across all retrieval strategies
- Benchmark latency and cost under load
- Write the ADRs (should mostly be filling in decisions already made)
- Polish deployment manifests; verify they apply cleanly to OpenShift AI
- Complete the runbook

---

## Completion Checklist

- [ ] Full query flow works end-to-end: user query → supervisor → research agent → LLM synthesis → SSE stream
- [ ] Ingestion KFP pipeline runs without errors on a 100+ document corpus
- [ ] LangSmith traces show all three agents executing with per-node timing
- [ ] LlamaGuard blocks at least one unsafe test query
- [ ] Redis semantic cache achieves >20% cache hit rate on the evaluation query set
- [ ] RAGAS evaluation report covers all 5 retrieval strategies with scores filled in
- [ ] At least 3 ADRs written with context, decision, and consequences sections completed
- [ ] `podman-compose up` starts all local services cleanly from a fresh state
- [ ] At least 3 OpenShift AI deployment manifests apply without errors
- [ ] Runbook covers the model update procedure end-to-end
