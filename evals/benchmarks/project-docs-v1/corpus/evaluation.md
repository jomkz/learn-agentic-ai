# Retrieval strategy evaluation status

No verified cross-strategy benchmark is included in this repository. The previous numerical
comparison could not be regenerated from the supplied code and has been withdrawn. It must
not be used to justify model promotion, architecture selection, or production quality claims.

The Phase 9 comparison now requires actual per-question outputs for each strategy, evaluated
against identical questions and reference answers. It applies the same evaluator to all runs;
there are no score bonuses or assumed latency/cost values.

```bash
uv run python -m mobility_ai.phase9.capstone \
  --run baseline=outputs/baseline.jsonl --run candidate=outputs/candidate.jsonl \
  --backend ragas --judge-model llama3.2 --embedding-model nomic-embed-text \
  --revision YOUR_COMMIT --output outputs/comparison.json
```

Install the `evaluation` extra and run Ollama before selecting `ragas`. Both judge and embedding
models are explicit; missing dependencies and judge failures stop the evaluation. The lexical
backend runs offline, but its exact-match and token-overlap metrics are **not faithfulness**.

Reports retain per-question answers, contexts, scores, dataset hashes, configuration, evaluator
version, and timestamps. Latency is reported only when captured in the input records; it is
mean total query latency, not p95 or time to first token. Local inference does not establish a
zero infrastructure cost, so unmeasured costs remain null.
