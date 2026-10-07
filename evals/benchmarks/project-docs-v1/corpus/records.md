# Evaluation inputs

`lexical-demo.jsonl` contains three hand-authored examples for exercising the report format.
The answers are fixtures, not captured model outputs or a held-out quality benchmark.
No latency or cost is invented for these records.

Run the offline format check:

```bash
uv run python -m mobility_ai.evals.ragas_harness \
  --eval-set evals/data/lexical-demo.jsonl --backend lexical \
  --output outputs/lexical-demo-report.json
```

For model evaluation, record one JSON object per question with `question`, `ground_truth`,
`contexts`, and the actual `answer`. Optional `latency_ms` is total retrieval and generation
wall time, not time to first token; `cost_usd` must come from actual usage and a dated price
configuration. Missing measurements remain null. Do not use training questions as held-out
quality evidence. Version the corpus, questions, model, prompts, and code revision with each run.

The capstone runbook describes how to capture outputs from a runnable pipeline.
