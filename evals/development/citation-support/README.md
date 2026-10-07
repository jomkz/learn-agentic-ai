# Citation-support development audit

These eleven inspected cases exercise the distinction between an existing citation ID and
support for an answer. They include recorded wrong-passage, extra-passage, invented-identifier,
and unsupported-reasoning failures, supported controls, a non-answer, a corrected citation,
and synthetic negation/distributed-evidence controls. Inputs retain all retrieved contexts;
the judge receives only the cited passage IDs/text, the question, and the actual answer.

```bash
ollama pull qwen2.5:7b
uv run python evals/development/citation-support/prepare.py \
  --output outputs/citation-support-suite
uv run python -m mobility_ai.evals.citation_support \
  --records outputs/citation-support-suite/records.jsonl \
  --judge-model qwen2.5:7b --output outputs/citation-support-audit
```

The preparation script writes input records, their origins/hashes, and provisional expected
labels separately. The audit never sends those labels or reference answers to the model.
Inspect `judgments.jsonl`, including raw judge responses, errors, quotes, and citation IDs.
`summary.json` counts outcomes across every input; it does not turn model verdicts into a
correctness score. Output directories must be new, preserving previous attempts.

The labels are Codex inspection, not independent human review. This is development data, not
the larger manually reviewed calibration set planned next. Preserve disagreements and invalid
judge outputs. Literal evidence and answer-excerpt checks establish provenance, not entailment.
The model can still miss unsupported reasoning, misread negation, or overlook a claim.

The audit operates on answer-level citation mappings. It checks overall answer support and
each citation's contribution, with an answer excerpt and source quote for supporting judgments.
It does not establish complete claim-level coverage or validate every inline citation placement.
It is an advisory diagnostic and does not change the capstone's acceptance behavior.
