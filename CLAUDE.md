# CLAUDE.md — Agentic AI & MLOps Learning Path

This file orients Claude Code to the repo structure and conventions so each session starts with full context.

## What this repo is

A structured 41-week learning path for building production-grade Agentic AI applications. The target stack is Python-first, open-source-first, with OpenShift AI as the production deployment platform. See [docs/index.md](docs/index.md) for the full curriculum.

## Repo structure

```
docs/
  index.md              — curriculum overview, phase table, dependency map
  resources.md          — books, courses, papers reading list
  phases/
    01-foundations.md   — one file per phase; this is where curriculum lives
    ...
    10-capstone.md

projects/               — hands-on code, one subdirectory per phase
  phase1-foundations/
  ...

evals/                  — reusable RAGAS evaluation harnesses (built in Phase 3, reused everywhere)
notebooks/              — Jupyter notebooks for exploration

pyproject.toml          — all Python dependencies (uv-managed, phase-gated extras)
podman-compose.yml      — local infrastructure (PostgreSQL+pgvector, Qdrant, Redis, Neo4j)
.env.example            — API key and service config template; copy to .env
```

## Phase convention

Each phase has three parts that must stay in sync:
1. `docs/phases/NN-name.md` — curriculum doc (objectives, key concepts, resources, projects, checklist)
2. `projects/phaseN-name/` — hands-on code directory
3. `pyproject.toml` `[project.optional-dependencies]` entry — phase-gated deps

When adding a new phase or technology to an existing phase, update all three.

## Dependency management

```bash
uv sync                          # base deps (Phase 1)
uv sync --extra langchain        # Phase 2
uv sync --extra rag              # Phase 3
uv sync --extra advanced-rag     # Phase 4
uv sync --extra agents           # Phases 5-6
uv sync --extra llamastack       # Phase 7
uv sync --extra ml               # Phase 8
uv sync --extra advanced         # Phase 9
uv sync --all-extras             # everything

uv add <package>                 # add a new dependency (updates pyproject.toml + uv.lock)
uv add --optional <group> <pkg>  # add to a phase-gated group
```

## Local services

```bash
podman-compose up -d postgres qdrant   # Phase 3+
podman-compose up -d redis             # Phase 4+
podman-compose up -d neo4j             # Phase 9
podman-compose up -d                   # everything
podman-compose down                    # stop, keep data
podman-compose down -v                 # stop, delete all volumes (clean slate)
```

Service URLs: PostgreSQL `localhost:5432`, Qdrant `http://localhost:6333`, Redis `localhost:6379`, Neo4j `http://localhost:7474`.

## Code quality

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy .                # type check
uv run pytest                # all tests
uv run pytest projects/phaseN-name/   # single phase
uv run jupyter lab           # notebooks
```

Config lives in `pyproject.toml` under `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`.

## Key technology decisions

| Concern | Choice | Note |
|---------|--------|------|
| Package manager | `uv` | Lockfile-based; faster than pip |
| Container runtime | Podman | Red Hat default; Docker Compose compatible |
| Local LLM | Ollama | No API key; OpenAI-compatible API |
| Default vector store | pgvector | Production-ready; in existing PostgreSQL |
| Agent orchestration | LangGraph | Stateful, checkpointed, multi-agent |
| Model serving (prod) | vLLM on OpenShift AI | PagedAttention; LoRA serving |
| Fine-tuning | QLoRA + TRL + Axolotl | PEFT-efficient; single consumer GPU |
| Prompt optimization | DSPy | Metric-driven; replaces hand-crafted prompts |
| Experiment tracking | MLflow + W&B | MLflow = registry; W&B = training visualization |
| Data versioning | DVC | Git-pointer to large files in S3/MinIO |
| Agent observability | LangSmith + OpenTelemetry | Dev tracing + production vendor-neutral |
| Drift monitoring | Evidently AI | Data drift + LLM output quality over time |
| Tool protocol | MCP + A2A | MCP: agent↔tool; A2A: agent↔agent |
| Security testing | Garak | LLM vulnerability scanning |

## Secrets

Never commit `.env`. Copy `.env.example` to `.env` and fill in keys. All API keys and service credentials are loaded via `python-dotenv`. Large model files (`*.gguf`, `*.safetensors`, `*.bin`) are gitignored.
