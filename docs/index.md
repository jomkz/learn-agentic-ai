# Agentic AI & MLOps on OpenShift AI — Learning Path

## Context

This learning path is for a senior software engineer/architect at Red Hat who needs to become proficient in designing, building, and leading the implementation of production-grade Agentic AI applications. The target stack is Python-first, open-source-first, and Red Hat/OpenShift-aligned.

**Goal activities:**
1. Architect, design, document, and develop Python apps using LangChain, LangGraph, and LlamaStack
2. Architect and lead implementation of Agentic AI applications
3. Architect scalable open-source ML solutions with distributed computing on OpenShift AI
4. Architect and design features using RAG, RAFT, GraphRAG, InstructLab, and their pipelines
5. Develop and optimize RAG pipelines

---

## Phases at a Glance

**Total duration: ~41 weeks (~10 months full-time / 12+ months at 20 hrs/week)**

| Phase | Topic | Duration | Cumulative |
|-------|-------|----------|------------|
| [1](phases/01-foundations.md) | Foundations + Python Modernization + Local Dev | 3 weeks | 3 weeks |
| [2](phases/02-langchain.md) | LangChain + LCEL + LangSmith + Streaming + Testing | 5 weeks | 8 weeks |
| [3](phases/03-rag-fundamentals.md) | RAG Fundamentals + Document Parsing + Vector Stores | 4 weeks | 12 weeks |
| [4](phases/04-advanced-rag.md) | Advanced RAG + Caching + Cost Optimization | 3 weeks | 15 weeks |
| [5](phases/05-langgraph.md) | LangGraph + Multi-Agent Workflows | 4 weeks | 19 weeks |
| [6](phases/06-agentic-patterns-mcp.md) | Agentic Patterns + MCP + Guardrails | 3 weeks | 22 weeks |
| [7](phases/07-llamastack.md) | LlamaStack | 3 weeks | 25 weeks |
| [8](phases/08-huggingface-openshift.md) | HuggingFace + LoRA/QLoRA + OpenShift AI + vLLM + KFP + Ray + MLOps | 8 weeks | 33 weeks |
| [9](phases/09-graphrag-raft-instructlab.md) | GraphRAG + RAFT + InstructLab + Multi-Modal | 6 weeks | 39 weeks |
| [Capstone](phases/10-capstone.md) | End-to-End Enterprise Platform | 2 weeks | 41 weeks |

---

## Dependency Map

```
Phase 1 (Foundations + Local Dev)
    ├── Phase 2 (LangChain + LCEL + Testing)
    │       ├── Phase 3 (RAG Fundamentals)
    │       │       └── Phase 4 (Advanced RAG + Caching)
    │       │               └── Phase 9 (GraphRAG/RAFT/InstructLab)
    │       └── Phase 5 (LangGraph)
    │               └── Phase 6 (Agentic Patterns + MCP + Guardrails)
    └── Phase 7 (LlamaStack) [can start after Phase 2]
Phase 8 (HuggingFace + OpenShift AI) [can start in parallel with Phase 5; needs Phase 3]
Phase 9 (GraphRAG/RAFT/InstructLab)  [needs Phases 4 and 8]
Capstone                              [needs all phases]
```

**Parallelism opportunity:** Once Phase 3 is complete, Phase 8's infrastructure track (OpenShift AI, vLLM, KFP, Ray) can run alongside Phases 5-7. This is a good split if two engineers are working together.

---

## Workspace Layout

```
.
├── docs/
│   ├── index.md              ← you are here
│   ├── resources.md          ← books, courses, channels, papers
│   └── phases/               ← one file per phase
├── projects/                 ← hands-on project code by phase
├── notebooks/                ← Jupyter notebooks for exploration
├── evals/                    ← reusable RAGAS evaluation harnesses
├── pyproject.toml            ← uv-managed dependencies (phase-gated extras)
└── podman-compose.yml        ← local infrastructure (pgvector, Qdrant, Redis, Neo4j)
```

---

## Key Technology Decisions

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Package manager | `uv` | Fast, lockfile-based, workspace-aware |
| Container runtime | Podman | Red Hat default; drop-in Docker-compatible |
| LLM API client | `openai` SDK | OpenAI-compatible; works with Ollama, vLLM, LlamaStack |
| Local model serving | Ollama | Zero-config; GGUF format; OpenAI-compatible API |
| Default vector store | pgvector | Production-ready; lives in existing PostgreSQL |
| Orchestration | LangGraph | Stateful, checkpointed, multi-agent |
| Model serving (prod) | vLLM on OpenShift AI | PagedAttention; OpenAI-compatible; GPU-efficient |
| Fine-tuning | QLoRA + TRL | Memory-efficient; works on single consumer GPU |
| Observability | LangSmith + MLflow | Tracing for agents; experiment tracking for training |

---

## Getting Started

```bash
# Install Phase 1 deps
uv sync

# Pull local models (no API key needed)
ollama pull llama3.2
ollama pull nomic-embed-text

# Start local services (needed from Phase 3)
podman-compose up -d postgres qdrant

# Open the first phase
open docs/phases/01-foundations.md
```

See the root [README](../README.md) for full prerequisites and quick-start instructions.

---

## Reference

- [Global Resources](resources.md) — books, courses, YouTube channels, papers reading list
- [README](../README.md) — prerequisites, quick start, service URLs
