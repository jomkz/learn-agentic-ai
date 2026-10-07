# Agentic AI & MLOps learning workspace

A Python learning path from LLM fundamentals to RAG, agents, and OpenShift AI.
Start with the [curriculum](docs/index.md) and [implementation status](docs/status.md).
Advanced deployment and training materials include exercises that require additional work;
the runnable local capstone is documented separately from the proposed enterprise architecture.

## Quick start — no API key or GPU required

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this repository:

```bash
uv python install 3.12
uv sync --locked
uv run python -m mobility_ai.capstone.app ingest \
  --corpus projects/capstone/data/corpus --db outputs/capstone.db --provider lexical
uv run python -m mobility_ai.capstone.app ask \
  --db outputs/capstone.db --question "What does pgvector add to PostgreSQL?"
uv run python -m mobility_ai.capstone.app evaluate \
  --db outputs/capstone.db --eval-set evals/data/capstone-questions.jsonl \
  --records outputs/local-run.jsonl --output outputs/local-report.json --backend lexical
```

This is a complete offline exercise: lexical count vectors are persisted in SQLite, cosine
search retrieves chunks, and extractive answers cite the source files. Lexical evaluation
measures token overlap, not factual faithfulness. SQLite uses a linear vector scan suitable
for small local corpora; pgvector remains a future production integration exercise.

For LLM-generated answers, install [Ollama](https://ollama.com), pull `llama3.2` and
`nomic-embed-text`, and ingest with `--provider ollama`. Re-ingest when switching embedding
models. See the [capstone runbook](projects/capstone/RUNBOOK.md).

Live CPU validation and a 24-question retrieval-depth benchmark are recorded in the
[September 23 results](evals/benchmarks/project-docs-v1/results/2026-09-23/README.md).
The benchmark preserves failed responses and separates application errors from judge scores.

## Dependencies by phase

Python 3.12 is the supported runtime, pinned in `.python-version`. The lockfile is committed.
Phase extras include their prerequisites: `rag` includes `langchain`, `advanced-rag` includes
`rag`, and `agents` includes `langchain`. `uv sync` selects the requested extras exactly;
repeat all extras that you want to retain when moving between independent tracks.

| Track | Install |
|---|---|
| Phase 1 and local capstone | `uv sync --locked` |
| Phase 2 | `uv sync --locked --extra langchain` |
| Phase 3 | `uv sync --locked --extra rag` |
| Phase 4, including DSPy | `uv sync --locked --extra advanced-rag` |
| Phases 5–6 | `uv sync --locked --extra agents` |
| Phase 7 | `uv sync --locked --extra llamastack` |
| Phase 8 training | `uv sync --locked --extra ml` |
| Phase 8 KFP compilation only | `uv sync --locked --extra pipelines` |
| Phase 9 graph integrations | `uv sync --locked --extra advanced` |
| RAGAS evaluation | `uv sync --locked --extra evaluation` |
| Notebooks | `uv sync --locked --extra notebooks --extra agents` |

GPU/ML extras are large. Install only the track you need; do not use `--all-extras` for the
core learning path. Optional cloud credentials can be configured with `cp .env.example .env`;
leave keys empty for local work. Tracing is disabled by default.

## Checks

```bash
uv sync --locked --extra agents --extra pipelines --extra evaluation
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pytest projects/phase3-rag/
```

Unit tests mock optional backends explicitly. The KFP compilation test runs when its extra
is installed. Service and GPU execution remain separate integration checks. CI tests the
core, phase installations, and the installed capstone outside the source tree. Heavyweight
extras can be checked through the CI workflow's manual `optional_phases` input.

The separate **Model security gate** workflow runs real Garak probes against an explicitly
configured Ollama endpoint and fails on missing/incomplete results or detected failures.
It is callable by a future promotion workflow; there is no automated model promotion here.

## Layout

```text
src/mobility_ai/phase1/ … phase9/  importable examples
src/mobility_ai/capstone/         runnable local application and component diagnostics
src/mobility_ai/evals/            explicit evaluation backends and security report gate
projects/phase*/tests/            phase tests
projects/capstone/               runbook, fixtures, deployment exercises, design records
evals/                           evaluation inputs and tests
docs/                            curriculum, status, and resources
notebooks/                       interactive exercises
```

Run modules with `uv run python -m mobility_ai.phaseN.module`; source files have moved from
`projects/` into the installed package. There are no phase-specific `sys.path` workarounds.

## Optional local services

Install Podman and `podman-compose` for the service exercises. Published ports bind to
localhost. Development credentials are examples; cluster credentials use Secrets.

```bash
podman-compose up -d postgres qdrant
podman-compose up -d redis
podman-compose up -d neo4j
podman-compose down
```

The offline capstone does not require these services. Redis response caching uses exact
queries and a namespace identifying the model, prompt, corpus version, and tenant. MCP file
tools default to `data/mcp`; create that directory or set `MCP_ALLOWED_ROOT` to an approved
folder. Absolute paths outside that folder, parent traversal, and symlinks are rejected.
