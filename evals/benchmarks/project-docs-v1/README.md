# Project documentation benchmark v1

This is a frozen, hand-authored task set: 8 repository documents, 20 answerable questions,
and 4 questions requiring abstention. Each answerable reference has an exact supporting
excerpt and source. The source manifest records where each snapshot came from and its hash.
The questions are separate from the four-question development smoke test. No training or
prompt tuning uses this set. It is a small local benchmark, not evidence of general RAG quality.

## Retrieval and generation protocol fixed before capture

- Compare cosine retrieval of **one passage versus three passages**. Use the same persisted
  corpus, embedding model, generation model, chunking (512 characters, 64 overlap), and prompt.
- Use `nomic-embed-text:v1.5` and `llama3.2:3b`. Record actual model digests and Ollama version.
- Generation uses temperature 0, seed 0, context 4096, and at most 512 output tokens. These
  settings improve repeatability but do not guarantee identical results across hardware/releases.
- Warm the model with a separate smoke prompt, then run all 24 questions in each configuration.
  Alternate configuration order for consecutive questions. This is one run, without significance claims.
- Preserve every attempted output, including citation-validation failures and server errors.
  Application errors remain in the 24-question denominator; do not replace rejected answers.
- Report complete reference-excerpt retrieval, exact abstentions on the four unknowns,
  false abstentions on answerable questions, application errors, and measured mean/median/p95
  latency. Excerpt matching is strict and can miss paraphrased or equivalent evidence elsewhere.
- Judge answerable, nonempty model outputs unless they match the exact abstention sentence,
  using RAGAS faithfulness, answer relevancy, and context precision. Alternate-wording
  abstentions remain eligible. Preserve judgment errors and report the scored denominator.
  RAGAS means are conditional on scorable outputs; they do not replace end-to-end failure rates.
- The local judge is the independent `qwen2.5:7b` model, with JSON requested by RAGAS
  prompts (without Ollama's forced JSON mode), seed 0, context
  8192, and a 1024-token limit. The locked RAGAS 0.3.1 overrides judge temperature to 1e-8 for single completions
  and 0.3 for multiple completions; the run records that policy. Treat its scores as fallible
  automated diagnostics. Inspect raw
  responses against references before drawing conclusions. Unmeasured costs remain null.

Pre-benchmark smoke testing found that the Llama 3B judge echoed the schema under forced
JSON mode and produced excessively long output in normal mode. Qwen 3B completed the metrics
but gave zero faithfulness to an obviously supported answer. The larger Qwen 7B judge is
selected using separate positive/negative smoke calibration; no benchmark outputs determine
the judge settings. The failed/cancelled smoke attempts are retained with the live-validation
artifacts; no benchmark questions were used to make this configuration choice.

## Reproduce

Start Ollama and pull `llama3.2:3b`, `nomic-embed-text:v1.5`, and `qwen2.5:7b`, then:

```bash
uv sync --locked --extra evaluation
uv run python -m mobility_ai.evals.benchmark capture \
  --output outputs/project-docs-v1
ollama stop llama3.2:3b
uv run python -m mobility_ai.evals.benchmark judge \
  --output outputs/project-docs-v1
```

Use `--ollama-url` and `--embedding-model` for either stage; `--generation-model` selects
the captured answer generator and `--judge-model` selects the evaluator. Capture
requires a new output directory and refuses to overwrite an existing run. Judging uses the
recorded outputs and refuses to overwrite previous judgments. Each request has a bounded
timeout; CPU evaluation may take tens of minutes. Stopping the unused generator lets the judge and embedder remain loaded on
servers configured for only two concurrent models. See Ollama's
[memory and concurrency settings](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests).

Inspect judgment errors before interpreting aggregates; a written summary is not itself a quality gate.

Artifacts include model/configuration metadata, hashes of the actual working-tree source,
the database, per-attempt JSONL with raw provider timing counters, lexical reports, operational
summaries, and individual RAGAS reports/errors. The latency p95 uses the nearest-rank method
on 24 sequential warm requests, includes failures, and measures client wall time.

The Ollama integration uses its documented [embedding endpoint](https://docs.ollama.com/api/embed).
The evaluator uses RAGAS's [explicit evaluation API](https://docs.ragas.io/en/v0.3.9/references/evaluate/).
