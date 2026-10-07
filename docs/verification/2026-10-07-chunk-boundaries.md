# Chunk-boundary verification — 2026-10-07

Follow-up priority 2 adds optional adjacent-chunk expansion to the existing SQLite workflow.
The inspected namespace failure was caused by a passage ending after the word `model`, before
the remaining identifiers. Expansion now supplies the complete source span without changing
ingestion or requiring a database migration. It remains disabled by default.

## Implementation and regression checks

Queries retain their top-k similarity matches and may add one neighbor on each side. The
application joins contiguous chunks from the same source using persisted offsets, removes
overlap, and records all contributing chunk indices and original matches. Joined passages
inherit the best original match's score and rank. Citations identify single chunks or ranges.
A configurable character budget retains every original match and skips neighbors that do
not fit; it fails explicitly when the original context alone is too large.

- 456 tests passed with 92.02% branch-inclusive coverage, above the 90% gate.
- Ruff lint/format passed for 110 Python files; mypy passed for 53 source files.
- Tests cover exact reconstruction, repeated/Unicode text, zero/high overlap, missing indices,
  document boundaries, ranking, neighbor limits, budgets, inconsistent stored text, reopening
  existing databases, citation ranges, CLI settings, and evaluation/benchmark provenance.
- A regression replays q18's original recorded passages. Expansion recovers the exact
  `overview.md#chunks-9-11` slice, including model, prompt, corpus version, and tenant.
- All five CI jobs passed for the implementation commit, including installed-package and
  isolated phase checks. Heavyweight optional phase jobs were not requested.
- All eight frozen corpus hashes, 30 September artifacts, and five structured-response
  artifacts from the previous follow-up remain unchanged.

## Live development comparison

The [reproduction instructions](../../evals/development/chunk-boundaries/README.md) select five
already inspected baseline questions: chunk settings, missing-database recovery, the namespace
list, and two unknowns. Both configurations use the same frozen eight-document corpus, 512/64
character chunking, `structured-rag-v2` prompt, generation settings, and model digests. Each
configuration captures top_k=1 and top_k=3. This is 20 attempts, with no retries or exclusions.

The local CPU instance used Ollama 0.34.3, `llama3.2:3b`, and `nomic-embed-text:v1.5`. It was
stopped after capture. Commit `9905a65` and recorded source hashes identify the implementation.
The character budget was 6000 for both configurations. Every returned passage was checked
against its exact source slice. Raw rejected responses are retained alongside their errors.

| Expansion / top-k | Full reference excerpt available | Application errors | Mean passage characters | Median query latency | Accepted explicit abstentions |
|---|---:|---:|---:|---:|---:|
| Off / 1 | 1/3 answerable | 1/5 | 427.2 | 3.45 s | 0/2 unknown |
| Off / 3 | 2/3 answerable | 1/5 | 1421.2 | 5.36 s | 0/2 unknown |
| On / 1 | 2/3 answerable | 1/5 | 1198.4 | 5.68 s | 0/2 unknown |
| On / 3 | 3/3 answerable | 2/5 | 3252.2 | 8.53 s | 0/2 unknown |

Excerpt availability is a strict retrieval diagnostic, not a correctness score. Latency includes
errors and covers sequential warm requests. These small sequential runs do not establish a
general performance or quality gain. No LLM judge ran; the case review is Codex inspection,
not independent human review. See the [per-case review and summary](../../evals/development/chunk-boundaries/results/2026-10-07/review.json),
[unexpanded records](../../evals/development/chunk-boundaries/results/2026-10-07/off/top3.jsonl),
and [expanded records](../../evals/development/chunk-boundaries/results/2026-10-07/on/top3.jsonl).
The result directories also contain top1 records, source/model metadata, and artifact hashes.

## Findings and adoption decision

- **Namespace evidence recovered:** at top3, expansion supplied the complete list and the model
  answered with all four identifiers. It cited architecture passage [3], while the supporting
  namespace list was in overview passage [2]. This is still a citation-support failure.
- **Ranking remains a limit:** at top1, the relevant overview document was absent in both
  modes. The expanded answer invented `model_id, task_id, timestamp` and cited a valid but
  unsupported passage. The unexpanded answer was rejected for nonexistent citation IDs.
- **Recovery instruction improved in one case:** top1 q08 changed from rollback advice to
  the correct `ingest` instruction after the neighboring runbook text was supplied.
- **Abstention remains unreliable on this corpus:** unexpanded top1 q21 invented throughput
  of `10`. Several other unknown outputs said information was absent but set `abstain=false`.
  Expanded q21 in both settings and q23 at top3 set `abstain=true` while including citations;
  the contract correctly rejected these contradictory responses.

Keep expansion opt-in. The implementation resolves the inspected source truncation, while the
live run exposes wrong citations, unsupported answers, inconsistent abstention decisions, and
added latency. Citation support is next, followed by judge calibration. Evaluate later changes
on a new held-out set before changing defaults or claiming broader quality improvements.
