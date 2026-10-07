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

## Expand passages across chunk boundaries

For evidence cut at a character boundary, enable adjacent-chunk expansion at query time:

```bash
uv run python -m mobility_ai.capstone.app ask \
  --db outputs/capstone.db --question "What does pgvector add to PostgreSQL?" \
  --top-k 3 --adjacent-chunks 1 --max-context-chars 6000
```

`--top-k` selects 1–20 similarity matches before expansion. `--adjacent-chunks 1` adds at most
one preceding and one following chunk from each match's source. Contiguous chunks are joined
using the stored chunk size and overlap, so repeated overlap text is removed and source text
is preserved exactly. Separate files and gaps in chunk indices are never joined. No re-ingestion
or database migration is required. Expansion defaults to 0; its quality benefit has not been
established on a new held-out set.

The 6000-character default budget applies to the total passage text in both modes. Every
original match is retained. If those matches alone exceed the budget, the query fails explicitly;
lower `--top-k` or raise `--max-context-chars`. Expansion tries neighbors in match order, next
then previous, and skips whole neighbors that do not fit. It never truncates a passage to fit.
The budget excludes the prompt and question and is not a token limit; account for the model's
context window separately.

Each returned passage lists `chunk_indices` for all contributing chunks and `matched_indices`
for its original similarity matches. `index` is the first contributing chunk. Its score and
position come from its best-ranked original match. Single-chunk citations retain
`source#chunk-N`; joined spans use `source#chunks-N-M`. These ranges identify the supplied
source span, not which individual sentence supports a claim.

The same flags work on `evaluate`; settings are recorded in report configuration. The benchmark
recorder also accepts the expansion and budget flags and records them per attempt and in
metadata. See the [off/on development comparison](../../evals/development/chunk-boundaries/README.md)
for the inspected cache-namespace example and controls.

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

## Audit citation support

Identifier validation only establishes that a cited passage was retrieved. To inspect evidence
support separately, audit saved evaluation JSONL or benchmark records with an explicit local
judge model:

```bash
ollama pull qwen2.5:7b
uv run python -m mobility_ai.evals.citation_support \
  --records outputs/local-run.jsonl --judge-model qwen2.5:7b \
  --output outputs/citation-support-audit
```

The audit reports `identifiers_valid` separately from the advisory `semantic_support` judgment.
It sends the question, actual answer, and cited passage text to the judge. Uncited passages,
source filenames, expected labels, and reference answers are excluded from its prompt.
Every cited ID must receive exactly one judgment. A supporting judgment must include an exact
answer excerpt and source quote; the application verifies both, allowing whitespace differences
but preserving case, punctuation, and negation. A quote found in a passage does not prove that
the passage entails the answer.

The model classifies whole-answer support and each citation's contribution independently, so
an otherwise supported answer can still have an irrelevant extra citation. It can also label
unsupported answers, uncertainty, and non-answers. These are uncalibrated diagnostics; they do
not reject or approve capstone answers and do not establish complete claim-level coverage or
correct inline citation placement.

Each non-blank input line remains in the output counts. Malformed records, oversized inputs,
judge failures, recorded abstentions, and rejected generations are reported explicitly.
The default `--max-input-chars 12000` bounds question, answer, and cited text without truncating
evidence; it is a character limit, not a guarantee about model token usage. Use a fresh output
directory. `judgments.jsonl` preserves raw judge responses and errors; `metadata.json` records
the input hash, model digest, source hashes, and prompt/settings. A failed final model-digest
check leaves partial records without a completed report.

See the [citation-support development cases](../../evals/development/citation-support/README.md)
for inspected failures and controls. Larger, independently reviewed judge calibration remains
the next step before treating these verdicts as a correctness gate.

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
- Context budget exceeded: lower `--top-k` or raise `--max-context-chars`; matches are never
  silently discarded to make room for neighbors.
- Inconsistent adjacent chunks: re-ingest the corpus; expansion refuses to join text whose
  overlap does not match the stored chunking configuration.
- Bad corpus update: preserve a copy of the previous SQLite database before replacing a
  corpus, or re-ingest the previous version of the source files. Do not copy while ingestion
  is writing; stop writers or use SQLite's backup API.

`mobility_ai.capstone.integration` checks importable symbols only. It is not a model-server,
database, or deployment health check. Training, promotion, monitoring dashboards, and
multi-service failover are optional exercises, not operational capabilities of this lab.
