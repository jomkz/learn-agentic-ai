# Live validation and project documentation benchmark — 2026-09-23

Live Ollama checks passed. The 24-question benchmark found that the existing three-passage
setting retrieved more reference evidence and produced fewer application rejections than
one passage, at roughly twice the median CPU latency. Both configurations still have quality
failures. These results support further development, not model promotion or production readiness.

## Setup and scope

- Ollama 0.34.3; Intel Core i7-11850H, 8 physical cores / 16 logical threads; CPU inference.
- Generator: `llama3.2:3b` (Q4_K_M); embeddings: `nomic-embed-text:v1.5` (768 dimensions).
- Independent judge: `qwen2.5:7b` (Q4_K_M), RAGAS 0.3.1.
- Eight frozen repository documents; 20 answerable questions and 4 unanswerable questions;
  one run per configuration. Chunk size 512 characters, overlap 64; cosine retrieval.
- Same corpus, generator, embedding model, generation prompt and settings in both runs;
  temperature 0, seed 0, context 4096, output limit 512 tokens. Model digests stayed unchanged.
- Warm generation, sequential requests, alternating configuration order. Latency is total
  client wall time, includes rejected answers, and uses nearest-rank p95. A judge-model download
  ran in the background during capture, so these are indicative workstation measurements,
  not an isolated load test or service SLO.
- Questions were authored before capture and were not used for training or prompt tuning.
  Judge selection used separate smoke calibration. This small hand-authored set is not a
  statistically representative or independently reviewed quality benchmark.

See the [frozen protocol](../../README.md), [source snapshot manifest](../../corpus-manifest.json),
[questions](../../questions.jsonl), and [run configuration, hashes and model digests](metadata.json).

## Live checks

The [smoke checks](live-validation/validation.json) verified 768-dimensional embeddings,
a cited pgvector answer, abstention on an unrelated question, idempotent ingestion, and
preservation of the existing corpus after a missing-model HTTP 404. The Ollama archive's
SHA256 matched the official release metadata; see [installation details](live-validation/installation.json).

Real SDK imports exposed two issues hidden by mocked tests: RAGAS imported adapters removed
in LangChain Community 0.4.2, and it required Pillow even for text-only prompts. The evaluation
extra and lockfile now install compatible dependencies; regular CI includes the actual SDK import.

Small-judge trials were retained: Llama 3B echoed a schema under forced JSON output and ran
long in normal mode; Qwen 3B scored a supported smoke answer at zero faithfulness. Qwen 7B
passed [basic positive/negative calibration](live-validation/judge-calibration.json), scoring
the supported answer at 1.0 and an invented email-billing claim at 0.0. Two calibration cases
are a sanity check, not proof of judge reliability.

## End-to-end results

| Measure | One passage | Three passages |
|---|---:|---:|
| Attempted questions | 24 | 24 |
| Answers rejected by the application | 9 / 24 | 2 / 24 |
| Complete reference-excerpt retrieval | 6 / 20 | 12 / 20 |
| Exact abstentions on unanswerable questions | 3 / 4 | 3 / 4 |
| Exact abstentions on answerable questions | 3 / 20 | 4 / 20 |
| Accepted non-abstaining answers | 9 | 15 |
| Mean latency | 2.70 s | 5.59 s |
| Median latency | 2.65 s | 4.99 s |
| p95 latency | 4.96 s | 8.60 s |

Reference-excerpt retrieval uses strict whitespace-normalized text matching, not semantic
recall. Equivalent evidence elsewhere may not count. Exact abstention counts intentionally
reflect the application's contract; alternate wording can be rejected despite expressing
uncertainty. Accepted citations verify source identifiers, not factual support.

## RAGAS diagnostics

The first pass scored 16 one-passage outputs and
16 three-passage outputs, with 1 and
0 judge errors respectively. The original records are preserved.
After fixing numerical score validation, only failed cases were replayed with the same
models and settings: 1 of 1 recovered. The reconciled results below
contain 17 and 16 scored outputs, with 0 and 0
remaining errors. Exact abstentions and
unanswerable cases were excluded from judging (7 and 8
records); their operational outcomes remain in the 24-question table above. Raw rejected
non-abstaining answers were still judged, but are counted as application failures.
Alternate-wording abstentions also remain in this diagnostic set: q01 with one passage
received faithfulness 1.0 but answer relevancy 0.0 despite providing no answer. A high
faithfulness score alone cannot establish usefulness or end-to-end correctness.
The judge also gave q13's three-passage answer faithfulness 1.0 despite its unsupported
claim that time to first token is a cost (see case inspection below). Passing the two smoke
calibration examples did not make this judge reliable enough to serve as a correctness gate.

The table below compares only the **14 question IDs with complete scores in both
configurations**, keeping the question set matched. These conditional scores do not account
for missing answers and do not represent an end-to-end success rate. The matched set also
excludes cases answered by only one configuration, including the q18 namespace failure;
inspect the full conditional aggregates and failure cases alongside it.

| Metric, matched scored questions | One passage | Three passages |
|---|---:|---:|
| faithfulness | 0.762 | 0.679 |
| answer_relevancy | 0.582 | 0.811 |
| context_precision | 0.429 | 0.649 |

See [paired results and question IDs](paired-judge-summary.json),
[reconciled conditional aggregates](judge-summary-reconciled.json),
[reconciled judgments](judgments-reconciled.jsonl),
[original individual judgments and errors](judgments.jsonl),
[original aggregates](judge-summary.json),
[replays with raw numerical scores](judge-retries.jsonl), and
[judge model/configuration metadata](judge-metadata.json).

The original evaluator rejected cosine roundoff above 1. A direct identical-question embedding
check reproduced `1.0000000000000002`. Evaluator version 2 allows a boundary tolerance of
1e-12, clips only that roundoff, and preserves legitimate negative cosine relevancy scores.
It still rejects non-finite or materially out-of-range values. Original successful scores
remain unchanged; recovered records carry their attempt number and original error. Replaying
a judge call can change its output, so these are separate attempts rather than reconstructed
first-pass scores. The raw replay scores show whether the same boundary issue recurred.
In this run, q17's replay reproduced answer relevancy `1.0000000000000007`; version 2
accepted it as 1.0. All three metrics completed, resolving the sole evaluation error.

A [provenance correction](provenance-correction.json) fixes an initially incorrect temperature
policy description; actual requests, answers and scores were not changed. The installed
RAGAS 0.3.1 wrapper uses 1e-8 for single completions and 0.3 for multiple completions.

## Failure cases and next work

- **Citation and abstention formatting:** q02 states the correct chunk sizes but omits citations
  under both configurations. Some abstentions use singular “document” or add “the question,”
  causing rejection. A structured answer/citation/abstention response is the next format improvement.
- **Chunk boundaries:** q18's retrieved passage cuts the Redis namespace list after “model.”
  The three-passage answer invents model/dataset names and versions. Test sentence-aware
  chunking or adjacent-chunk expansion on development questions before a new held-out evaluation.
- **Unsupported reasoning:** q13's three-passage answer misclassifies time to first token as
  a cost despite retrieving the latency definition. A syntactically valid citation did not prevent this.
- **Wrong passage cited:** q08's three-passage answer correctly recommends `ingest` but cites
  passage [1]; the supporting instruction is in [2]. Checking that a citation identifier exists
  does not verify that the cited passage supports the answer.
- **Regressions with more context:** q06 and q09 were answered in the one-passage run but
  became exact abstentions in the three-passage run. More context is not uniformly beneficial.

These are Codex inspections of the saved passages and answers, not independent human labels.
The examples and their limits are recorded in [case inspection](case-review.json). Retain this
run as the baseline and use a new held-out set after developing improvements. No GraphRAG,
RAFT, InstructLab, GPU, cluster, security-scan, or production-capacity comparison was performed.

## Artifacts and verification

[One-passage raw answers](top1.jsonl) · [Three-passage raw answers](top3.jsonl) ·
[Operational summary](summary.json) · [One-passage lexical diagnostics](top1-lexical.json) ·
[Three-passage lexical diagnostics](top3-lexical.json) · [Regression checks](verification.json)

All 402 tests passed with 91.55% coverage. Ruff, formatting, mypy, and diff checks passed.
Model costs were not measured and remain null. The temporary server was
[stopped after the run](live-validation/teardown.json);
its binaries/models remain in `/tmp/mobility-ollama` and `/tmp/mobility-ollama-models` for reuse.
