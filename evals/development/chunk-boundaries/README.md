# Chunk-boundary development comparison

Compare unexpanded retrieval with one adjacent chunk on each side, using the inspected
cache-namespace failure (q18) and four controls from the September corpus. This is development
data, not a new held-out benchmark. The prompt, model settings, chunking, and embeddings stay
the same across configurations. Expansion is optional and disabled by default.

The preparation script verifies and copies the frozen corpus into a fresh output directory.
It selects q02 (chunk settings), q08 (missing database), q18 (namespace list), and two unknowns
(q21 and q23). Run from the repository root with local Ollama and the two baseline models:

```bash
uv run python evals/development/chunk-boundaries/prepare.py \
  --output outputs/chunk-boundary-suite
uv run python -m mobility_ai.evals.benchmark capture \
  --suite outputs/chunk-boundary-suite --output outputs/chunk-boundaries-off \
  --adjacent-chunks 0 --max-context-chars 6000
uv run python -m mobility_ai.evals.benchmark capture \
  --suite outputs/chunk-boundary-suite --output outputs/chunk-boundaries-on \
  --adjacent-chunks 1 --max-context-chars 6000
```

Each configuration records all five questions with top_k=1 and top_k=3. Inspect every output,
including errors and abstentions. Check whether the complete namespace list reaches the model,
whether the answer actually uses it, whether citations point to the supporting span, and the
extra context/latency cost. Do not treat identifier validity or lexical overlap as claim support.
The sequential runs are a development comparison, not a controlled throughput experiment.

The offline regression additionally replays q18's original retrieved passages through context
expansion. It verifies the exact source slice and provenance independently of a fresh embedding
ranking or model answer. Neither the frozen corpus nor its original results are rewritten.

The [October 7 verification](../../../docs/verification/2026-10-07-chunk-boundaries.md) records
all 20 attempts, improved excerpt availability, added latency, and the remaining citation,
answer, and abstention failures. The results support keeping expansion opt-in.
