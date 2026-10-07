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
an `abstained` boolean, retrieved passages, total query latency, and generation mode. The
offline mode quotes retrieved text; it is not an LLM. Queries with no matching evidence return an explicit
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

The `structured-rag-v2` prompt uses Ollama's
[JSON schema output format](https://docs.ollama.com/capabilities/structured-outputs).
The model returns `answer` text, a `citations` list of unique positive integer document IDs,
and an `abstain` boolean. The application checks the schema and retrieved IDs, then appends
the citation markers to the answer. The model must leave bracketed citation markers out of
the answer text. A non-abstaining answer needs text and at least one citation. An abstention
must have no citations; the application renders the canonical abstention message regardless
of the model's wording. Malformed or contradictory responses fail explicitly.

Citation markers refer to the answer as a whole. Valid IDs do not prove that a passage supports
each claim. The [development checks](../../evals/development/structured-responses/README.md)
exercise this contract with a real local model.

## Capture and evaluate outputs

```bash
uv run python -m mobility_ai.capstone.app evaluate \
  --db outputs/capstone.db --eval-set evals/data/capstone-questions.jsonl \
  --records outputs/local-run.jsonl --output outputs/local-report.json \
  --backend lexical --revision YOUR_COMMIT
```

The four-question fixture checks the workflow, including an unanswerable question. It is
not a production quality benchmark. `records` retains actual answers, retrieved contexts,
references, citation mappings, abstention state, and measured total query latency. The report
includes per-question scores, configuration, timestamps, and hashes. Missing cost is null,
not assumed zero.
Legacy records still load with null citation/abstention metadata. Evaluator version 3 includes
the added metadata in dataset hashes; hashes from earlier evaluator versions are not comparable.

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

## Reproduce the live documentation benchmark

The [frozen benchmark](../../evals/benchmarks/project-docs-v1/README.md) compares one versus
three retrieved passages on 20 answerable and 4 unanswerable questions. It records every
attempt, model digests, source hashes, latency, and judgment errors. Use a fresh output directory:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5
ollama pull qwen2.5:7b
uv sync --locked --extra evaluation
uv run python -m mobility_ai.evals.benchmark capture --output outputs/documentation-benchmark
ollama stop llama3.2:3b
uv run python -m mobility_ai.evals.benchmark judge --output outputs/documentation-benchmark
```

The answer generator and judge are separate models. CPU judging can take tens of minutes.
The [recorded run](../../evals/benchmarks/project-docs-v1/results/2026-09-23/README.md) includes
known failure cases; these small-sample results do not establish production readiness.
The September artifacts remain frozen. Running the current code captures the new structured
contract rather than reproducing the original prompt; use the recorded source revision for
historical reproduction. New captures retain `raw_response` even on validation failure. Rejected
payloads have an empty scored answer and null citation/abstention state; they remain in the
error and latency denominators and are excluded from LLM judging. Historical records retain
their original scoring behavior.

## Troubleshooting and rollback

- Missing database: run `ingest`; `ask` never silently creates a new corpus.
- Ollama connection/model error: check the endpoint and `ollama list`, then repeat the command.
- Invalid response schema or citation: inspect the benchmark's `raw_response` and the prompt/model;
  the command fails. Plain-text model responses are no longer accepted by the Ollama adapter.
- Invalid vector dimensions: re-ingest with the intended embedding model.
- Bad corpus update: preserve a copy of the previous SQLite database before replacing a
  corpus, or re-ingest the previous version of the source files. Do not copy while ingestion
  is writing; stop writers or use SQLite's backup API.

`mobility_ai.capstone.integration` checks importable symbols only. It is not a model-server,
database, or deployment health check. Training, promotion, monitoring dashboards, and
multi-service failover are optional exercises, not operational capabilities of this lab.
