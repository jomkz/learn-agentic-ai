# Structured response verification — 2026-10-07

Follow-up priority 1 replaces free-text inference with explicit answer, citation, and
abstention fields. The application renders citation markers from validated IDs and uses the
abstention flag to select its canonical message. Citation validation still checks identifiers,
not support for individual claims.

## Regression checks

- 433 tests passed with 91.70% branch-inclusive coverage, above the 90% gate.
- Ruff lint and formatting passed; mypy passed for all 52 source files.
- Tests cover missing citations, alternate abstention wording, malformed JSON, wrong types,
  missing/extra fields, duplicate/out-of-range IDs, contradictory state, raw rejected-response
  preservation, and loading historical evaluation records.
- All eight frozen corpus hashes and 30 September artifact hashes still match.
- All five CI jobs passed for the implementation commit: the main checks, installed-package
  workflow, and isolated LangChain, agents, and pipeline environments. Heavyweight optional
  phase checks were not requested.

## Live development check

The [four-question fixture](../../evals/development/structured-responses/README.md) ran once
with one and three requested passages on a temporary CPU instance of Ollama 0.34.3. The corpus
contains only two chunks, so the latter setting supplies at most two passages. The generator
was `llama3.2:3b`; embeddings used `nomic-embed-text:v1.5`. Both model digests match the September
baseline. The official Ollama archive's SHA256 was verified before inference. The service was
stopped after capture.

| Check | Result |
|---|---|
| Response schema and retrieved citation IDs | 8/8 valid; zero application errors |
| Answerable questions | 4/4 answered with the reference facts and the supporting source |
| Unknown questions | 4/4 explicitly abstained, with no citations |
| Exact source selection on answerable questions | 3/4 matched; one answer also cited an irrelevant source |

The top3 chunk-defaults answer returned `512, 64 [1] [2]`. Source 1 contains the chunk settings;
source 2 describes pgvector and does not support that answer. A strict source-set check failed
on this case. The output was preserved without retry or filtering and is an additional
development case for citation-support work. The contract check passed; semantic citation
support did not pass every case.

The run is an inspected development check, not a held-out quality benchmark. No LLM judge ran.
See the [per-case review](../../evals/development/structured-responses/results/2026-10-07/validation.json),
[top1 records](../../evals/development/structured-responses/results/2026-10-07/top1.jsonl),
[top3 records](../../evals/development/structured-responses/results/2026-10-07/top3.jsonl), and
[provenance](../../evals/development/structured-responses/results/2026-10-07/metadata.json).
The records retain raw provider JSON, rendered answers, retrieved text, citation mappings,
latency, and provider counters. Source hashes and commit `a7663d3` identify the tested implementation.

## Compatibility and remaining work

Ollama responses must now satisfy `structured-rag-v2`; plain-text provider responses fail.
The CLI retains its answer and citation mapping fields and adds `abstained`. Evaluation records
retain both citation mappings and abstention state; older records load with null metadata.
Evaluator version 3 includes those fields in dataset hashes. RAGAS receives only its original
question, reference, context, and answer fields.

New benchmark captures keep rejected provider content in `raw_response`, with an empty scored
answer and null citation/abstention state. All attempts remain in error and latency counts;
invalid responses are excluded from LLM judging. Historical saved records and results remain
unchanged. The known uncalibrated-judge limitations still apply.

The next priority is chunk boundaries, followed by citation support and judge calibration.
Develop those changes against inspected failures and evaluate broader quality on a new
held-out set.
