# Local knowledge assistant runbook

Status: runnable local application; see [deployment exercises](deploy/README.md) for KFP and
OpenShift prerequisites. This runbook only lists commands implemented in the repository.

## Install and ingest

```bash
uv python install 3.12
uv sync --locked
uv run python -m mobility_ai.capstone.app ingest \
  --corpus projects/capstone/data/corpus --db outputs/capstone.db --provider lexical
```

Expected result: JSON reporting three chunks and `provider: lexical`. The corpus consists of
UTF-8 `.txt` or `.md` files; a single file is also accepted. Defaults are 512 **characters** per
chunk and 64 characters overlap. Ingestion replaces the corpus atomically, removes deleted
sources, and does not duplicate chunks when repeated. Embedding failures preserve the old
corpus. SQLite vectors are for this local lab, not a scalable production vector index.

## Query and inspect sources

```bash
uv run python -m mobility_ai.capstone.app ask \
  --db outputs/capstone.db --question "What does pgvector add to PostgreSQL?"
```

The response includes the answer, numbered citations mapped to source/chunk identifiers,
retrieved passages, total query latency, and generation mode. The offline mode quotes
retrieved text; it is not an LLM. Queries with no matching evidence return an explicit
abstention. Citation validation checks identifiers, not whether every generated claim is true.

## Run with Ollama

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
uv run python -m mobility_ai.capstone.app ingest \
  --corpus projects/capstone/data/corpus --db outputs/ollama.db --provider ollama
uv run python -m mobility_ai.capstone.app ask \
  --db outputs/ollama.db --question "What does pgvector add to PostgreSQL?"
```

Ollama must be running. Use `--ollama-url` on each command for a different endpoint. The
embedding and generation model identifiers are saved with the corpus; `ingest` accepts
`--embedding-model` and `--generation-model`. Use immutable model tags/digests for reproducible
experiments and re-ingest after an embedding change. Server failures are surfaced, not hidden
by fabricated answers or a silent change of provider. Local LLM execution requires enough RAM
for the chosen models; a GPU is optional and affects latency.

## Capture and evaluate outputs

```bash
uv run python -m mobility_ai.capstone.app evaluate \
  --db outputs/capstone.db --eval-set evals/data/capstone-questions.jsonl \
  --records outputs/local-run.jsonl --output outputs/local-report.json \
  --backend lexical --revision YOUR_COMMIT
```

The four-question fixture checks the workflow, including an unanswerable question. It is
not a production quality benchmark. `records` retains actual answers, retrieved contexts,
references, and measured total query latency. The report includes per-question scores,
configuration, timestamps, and hashes. Missing cost is null, not assumed zero.

For RAGAS judging of recorded model outputs:

```bash
uv sync --locked --extra evaluation
uv run python -m mobility_ai.evals.ragas_harness \
  --eval-set outputs/local-run.jsonl --backend ragas \
  --judge-model llama3.2 --embedding-model nomic-embed-text \
  --revision YOUR_COMMIT --output outputs/ragas-report.json
```

Use separate training and evaluation questions for actual model comparisons. Configure both
judge and embedder explicitly. RAGAS failures stop the run. See the
[evaluation status](evaluation/ragas_comparison.md) before drawing strategy conclusions.

## Troubleshooting and rollback

- Missing database: run `ingest`; `ask` never silently creates a new corpus.
- Ollama connection/model error: check the endpoint and `ollama list`, then repeat the command.
- Missing/invalid citation: inspect the captured output and prompt/model; the command fails.
- Invalid vector dimensions: re-ingest with the intended embedding model.
- Bad corpus update: preserve a copy of the previous SQLite database before replacing a
  corpus, or re-ingest the previous version of the source files. Do not copy while ingestion
  is writing; stop writers or use SQLite's backup API.

`mobility_ai.capstone.integration` checks importable symbols only. It is not a model-server,
database, or deployment health check. Training, promotion, monitoring dashboards, and
multi-service failover are optional exercises, not operational capabilities of this lab.
