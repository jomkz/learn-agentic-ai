# Citation-support verification — 2026-10-07

Follow-up priority 3 adds an advisory audit for recorded answers. Identifier checks establish
that a citation maps to a retrieved passage; the model separately assesses whole-answer
support and each citation's contribution. Capstone answer acceptance remains unchanged.

## Implementation and regression checks

The audit accepts canonical evaluation records and legacy/current benchmark records, without
guessing identifiers from prose. Only the question, actual answer, and cited passage IDs/text
reach the judge. References, provisional expected labels, filenames, and uncited contexts are
excluded. Every supplied ID must be assessed exactly once. Supporting judgments require a
literal answer excerpt and a quote from that same passage, normalizing only whitespace.

Invalid records, explicit abstentions, rejected generations, oversized inputs, and judge errors
remain in the output counts. Malformed judgments never become support verdicts. Raw responses,
input/source hashes, model digest, settings, and timestamps are retained in a fresh directory.

- 503 tests passed with 92.75% branch-inclusive coverage, above the 90% gate.
- Ruff lint/format passed for 113 Python files; mypy passed for 54 source files.
- Tests cover prompt isolation, identifier preservation, irrelevant extra citations, literal
  claim/quote provenance, partial/error responses, exclusions, input bounds, model changes,
  record compatibility, output accounting, and CLI behavior.
- All five CI jobs passed for the implementation revision, including installed-package and
  isolated phase checks. Heavyweight optional phase jobs were not requested.
- All eight frozen corpus hashes, 30 September artifacts, five structured-response artifacts,
  and nine chunk-boundary artifacts remain unchanged.

## Live development audit

The [eleven-case suite](../../evals/development/citation-support/README.md) includes recorded
wrong/extra citations, unsupported reasoning, supported controls, a non-answer, a controlled
citation correction, and synthetic negation/distributed-evidence cases. The expected labels
are provisional Codex inspection, not independent human review or held-out calibration.
Original records and transformations are preserved with their hashes.

Both attempts use the same inputs and local CPU Ollama 0.34.3 with `qwen2.5:7b`, digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
The judge uses temperature 0, seed 0, an 8192-token context, and a 1024-token output limit.
There are no selective retries or excluded cases. Both complete attempts are archived.

The first attempt used `citation-support-v1` at commit `dfa8e92`. Four judgments failed the
contract: the extra-passage case omitted a citation, while vector search, negation, and
distributed evidence copied or paraphrased source text as an answer excerpt. The validator
retained these as judge errors. Version 2 at commit `1507292` constrains the native JSON schema
to the supplied citation count and IDs, and emphasizes verbatim copying from the answer.
It retains the same strict post-response checks.

| Attempt | Inputs | Valid judgments | Judge errors | Overall labels agreeing with provisional expectations | Overall label disagreements |
|---|---:|---:|---:|---:|---:|
| v1 | 11 | 7 | 4 | 5 | 2 |
| v2 | 11 | 8 | 3 | 6 | 2 |

Version 2 assessed both passages in the extra-citation case and flagged the irrelevant one.
The vector-search, negation, and distributed-evidence cases still failed literal answer-excerpt
validation. Their raw labels are retained but are not counted as valid judgments. The two
valid label disagreements in each attempt are the throughput non-answer and the supported
namespace control. These counts describe inspected cases, not a judge accuracy estimate.

See the [per-case comparison](../../evals/development/citation-support/results/2026-10-07/review.json),
[v1 judgments](../../evals/development/citation-support/results/2026-10-07/v1/judgments.jsonl),
and [v2 judgments](../../evals/development/citation-support/results/2026-10-07/v2/judgments.jsonl).
The archive includes the shared input, provenance, provisional labels, per-attempt metadata
and summaries, and SHA-256 artifact manifest. All 54 recorded source hashes match the
corresponding implementation revision. The temporary Ollama service was stopped after capture.

## Interpretation and next step

The first attempt also rejected the correctly cited namespace list while quoting the sentence
that explicitly supports it. It classified the throughput non-answer as unsupported. For the
latency reasoning case, its unsupported label matched the provisional expectation, but its
explanation falsely said the passage did not mention p95/time to first token; it also rejected
a cost claim copied directly from the source. Matching an overall label does not validate the
reasoning or individual citation judgments.

Version 2 repeated the latency rationale error and throughput misclassification. Its namespace
rejection introduced a requirement that the listed identifiers be the *only* required ones,
although neither the question nor the answer made that exclusivity claim. Tightening the
response contract did not resolve these semantic judgment failures.

This audit supplies inspectable evidence, not a correctness gate or an ALCE metric. Exact
quotes establish provenance, not entailment. Answer-level mappings do not establish exhaustive
claim coverage or validate every inline citation placement. The next priority is a larger,
independently reviewed calibration set with supported and unsupported claims, non-answers,
wrong/extra citations, negation, and reasoning failures. Keep validator errors in the
denominator and review rationales as well as labels. Reserve a new held-out question set for
later quality evaluation before changing generation or retrieval defaults.
