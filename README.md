# Agentic AI & MLOps on OpenShift AI — Learning Path Workspace

Structured workspace for the Agentic AI & MLOps on OpenShift AI learning path.
See [docs/index.md](docs/index.md) for the full curriculum.

## Prerequisites

Install these before starting Phase 1.

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.11+** | Runtime | `dnf install python3.11` or https://python.org |
| **uv** | Package manager (replaces pip/venv) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Ollama** | Local LLM serving — no API key needed for dev | https://ollama.com |
| **Podman** | Container runtime (Red Hat default) | `dnf install podman podman-compose` |
| **Git** | Version control | `dnf install git` |

Optional (cloud API access — needed for some exercises):
- Anthropic API key: https://console.anthropic.com
- OpenAI API key: https://platform.openai.com
- LangSmith API key (free): https://smith.langchain.com

## Quick Start

```bash
# 1. Clone and enter the workspace
git clone <repo-url> && cd learn-agentic-ai

# 2. Copy and fill in API keys
cp .env.example .env
# Edit .env — at minimum, set ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Pull a local model (no API key needed)
ollama pull llama3.2          # 3B — fast, good for Phase 1 iteration
ollama pull llama3.1:8b       # 8B — better quality, needs ~8GB RAM
ollama pull nomic-embed-text  # embedding model for RAG phases

# 4. Create Python environment and install Phase 1 dependencies
uv sync

# 5. Start local infrastructure services (needed from Phase 3 onward)
podman-compose up -d

# 6. Launch Jupyter for notebook-based experimentation
uv run jupyter lab
```

## Installing Dependencies by Phase

Dependencies are organized as optional groups in `pyproject.toml` to avoid installing
everything upfront. Install each group when you reach that phase.

```bash
# Phase 2 — LangChain, FastAPI, streaming
uv sync --extra langchain

# Phase 3 — RAG: Docling, pgvector, Qdrant, RAGAS
uv sync --extra rag

# Phase 4 — Advanced RAG: Redis semantic cache, rerankers, DSPy
uv sync --extra advanced-rag

# Phase 5-6 — LangGraph, MCP, A2A, Guardrails, Garak, OpenTelemetry
uv sync --extra agents

# Phase 7 — LlamaStack
uv sync --extra llamastack

# Phase 8 — HuggingFace, LoRA/QLoRA, Axolotl, MLflow, W&B, DVC, Ray, KFP
uv sync --extra ml

# Phase 9 — GraphRAG, Neo4j
uv sync --extra advanced

# Everything at once (not recommended for Phase 1)
uv sync --all-extras
```

## Local Infrastructure Services

`podman-compose.yml` runs all the data services needed across the learning path.
You do not need all of them immediately — start only what the current phase requires.

```bash
# Start everything
podman-compose up -d

# Start only what Phase 3 needs
podman-compose up -d postgres qdrant

# Start only what Phase 4 adds
podman-compose up -d redis

# Phase 9 adds Neo4j
podman-compose up -d neo4j

# Check status
podman-compose ps

# View logs
podman-compose logs -f postgres

# Stop everything, preserve data
podman-compose down

# Stop everything, delete all data volumes (clean slate)
podman-compose down -v
```

### Service URLs

| Service | URL | Notes |
|---------|-----|-------|
| PostgreSQL + pgvector | `localhost:5432` | DB: `ragdb`, User: `postgres`, Password: `postgres` |
| Qdrant | http://localhost:6333 | Dashboard at http://localhost:6333/dashboard |
| Redis | `localhost:6379` | RedisInsight UI at http://localhost:8001 |
| Neo4j Browser | http://localhost:7474 | Bolt: `localhost:7687`, Password: `password` |

## Project Structure

.
├── README.md                         # This file
├── docs/index.md                     # Full curriculum — start here
├── pyproject.toml                    # Python dependencies (uv)
├── podman-compose.yml                # Local infrastructure services
├── .env.example                      # API key template → copy to .env
├── .gitignore
│
├── projects/
│   ├── phase1-foundations/           # Pydantic models, async LLM client, prompt engineering
│   ├── phase2-langchain/             # Research assistant agent, streaming API, test suite
│   ├── phase3-rag/                   # Technical documentation search (Docling + pgvector)
│   ├── phase4-advanced-rag/          # Advanced retrieval + semantic cache + cost optimization
│   ├── phase5-langgraph/             # Autonomous research pipeline (multi-agent)
│   ├── phase6-mcp-guardrails/        # MCP server + guardrailed agent
│   ├── phase7-llamastack/            # Portable AI application
│   ├── phase8-openshift/             # Production RAG platform on OpenShift AI
│   └── phase9-domain-adaptive/       # Domain-adaptive knowledge system + ADR
│
├── notebooks/                        # Jupyter notebooks for exploration and experiments
└── evals/                            # Reusable RAGAS evaluation harnesses (built in Phase 3)
```

## Running Tests

```bash
# All tests
uv run pytest

# Tests for a specific phase
uv run pytest projects/phase2-langchain/

# With verbose output
uv run pytest -v
```

## Key Decisions and Conventions

- **Ollama first**: all exercises are designed to run locally with Ollama before using cloud APIs — avoid cost and latency during iteration
- **Podman over Docker**: Red Hat default; all compose files use Podman; `docker compose` also works as a drop-in
- **pgvector as default vector store**: use pgvector for anything that needs to persist; use Chroma only for throwaway experiments in early phases
- **`.env` for secrets**: never hardcode API keys; always load with `python-dotenv`
- **`uv` over pip**: faster, reproducible, lockfile-based; use `uv add <package>` to add new dependencies
