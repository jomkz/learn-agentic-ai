# Structured response development checks

These four deliberately simple, inspected questions exercise the response contract. The
chunk-size case targets the uncited-answer failure seen in the September benchmark; two
questions require abstention. They are development inputs, not a held-out quality benchmark.
Unit tests separately inject alternate abstention wording, missing citations, malformed JSON,
invalid citation IDs, and contradictory state.

With a local Ollama server and the two models available, capture all four questions with one
and three requested passages (the corpus has only two chunks):

```bash
uv run python -m mobility_ai.evals.benchmark capture \
  --suite evals/development/structured-responses \
  --generation-model llama3.2:3b --embedding-model nomic-embed-text:v1.5 \
  --output outputs/structured-response-check
```

Inspect every record: answerable cases should have `status: ok`, `abstained: false`, and the
correct source citation; unknown cases should have `status: ok`, `abstained: true`, no citations,
and the canonical abstention message. Check that the chunk answer retains both 512 and 64
and that the pgvector answer describes vector similarity search. The recorder preserves raw
JSON, retrieved passages, model digests, source hashes, and timings. Lexical metrics are
diagnostics only; no judge or broad quality claim is part of this check.

The [October 7 verification](../../../docs/verification/2026-10-07-structured-responses.md)
records eight valid structured responses and one citation-support failure: the top3 chunk-size
answer also cited the unrelated pgvector passage. That result is retained for the later
citation-support follow-up. Passing the response contract does not imply correct source selection.
