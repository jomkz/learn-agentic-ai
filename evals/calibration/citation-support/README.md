# Independent citation-judge calibration

**Status: tooling and unlabelled packet ready; independent review deferred.** No completed
human labels, new live judge run, or calibration result exists for this set yet. The judge
remains advisory. Resume from the saved packet whenever review time is available.

## Review packet

Open [suite/packet/review.html](suite/packet/review.html) in a local browser. It contains 32
cases and 38 cited passages, with no preselected labels or judge predictions. The browser
shows the same evidence as the judge: question, actual answer, and cited passage IDs/text.
It excludes source filenames, references, and uncited passages.

Label whole-answer support and each citation's contribution, then explain the evidence and
reasoning. The page includes the labeling guide. **Save draft** downloads progress; use
**Load draft** to resume. Nothing is automatically persisted when the page closes, and the
page makes no network requests. **Export completed labels** requires every label, an
explanation for each case, a reviewer identifier, and confirmation that the review was done
without consulting judge predictions or suggested labels. That confirmation is an
attestation, not something the software can independently prove.

Save the completed download as `outputs/judge-calibration/labels.reviewed.json`. The JSON
template is also available for text-editor review. Keep completed labels separate from the
blank template and freeze them before running or inspecting the judge. Do not use an LLM to
fill the independent review. Prior exposure to an anchor case should be noted in its explanation.

The packet mixes eleven previously inspected development examples with 21 new authored
controls covering units, quantities, negation, list scope, non-answers, qualified claims,
conflicting evidence, distributed/uncited evidence, extra citations, and unsupported reasoning.
These are calibration examples, not a new held-out benchmark or a representative random
sample. Authored controls are not newly generated model answers. Label proportions will only
be known after human review; no suggested labels are supplied.

`records.jsonl` is the exact judge input. `origins.json` preserves provenance and transformations;
case IDs and ordering in the packet do not reveal those origins. The five suite artifacts are
hashed in `artifacts.sha256.json`. Preserve this set and the earlier frozen benchmark.

## Resume after independent labels are saved

Use the existing citation judge unchanged for the first comparison. Do not tune its prompt
against these labels before capturing that run. Every judge error stays in the denominators.

```bash
ollama pull qwen2.5:7b
uv run python -m mobility_ai.evals.citation_support \
  --records evals/calibration/citation-support/suite/records.jsonl \
  --judge-model qwen2.5:7b --output outputs/judge-calibration/audit-v2
uv run python -m mobility_ai.evals.calibration compare \
  --records evals/calibration/citation-support/suite/records.jsonl \
  --labels outputs/judge-calibration/labels.reviewed.json \
  --audit outputs/judge-calibration/audit-v2 \
  --output outputs/judge-calibration/comparison-v2
```

The comparison rejects incomplete reviews, duplicate/missing cases or citations, mismatched
input hashes, incomplete audits, changed recorded answers/passages, and saved judgments that
differ from their raw responses. It reports overall agreement across all cases/citations,
conditional agreement among valid judgments, confusion counts, disagreements, and unassessed
IDs. A zero denominator produces `null`, not a perfect score. Output directories must be new.

Inspect **every** entry in `cases.json`, including agreements: a correct label can have a faulty
rationale. Its `rationale_review` and `rationale_notes` fields start empty for this second
review. Record whether the rationale is acceptable, faulty, or uncertain, and explain issues.
Keep the original comparison and save annotations separately. The tool never automatically
marks rationale review complete or promotes the judge into a correctness gate. Resolve
ambiguities with further independent review before adoption; agreement with one reviewer is
not established accuracy. Later answer-quality claims still need a new held-out question set.

## Reproduction and reuse

```bash
uv run python evals/calibration/citation-support/prepare.py \
  --output outputs/judge-calibration/reproduced-suite
uv run python -m mobility_ai.evals.calibration prepare \
  --records path/to/records.jsonl --output outputs/another-review-packet
```

The general packet builder accepts uniquely identified, non-abstaining recorded answers with
valid citation mappings. Explicit abstentions and rejected generations have no support verdict
in the citation auditor and belong in separate generation/abstention checks. Judge service,
output-contract, and input-budget failures on eligible cases remain visible in calibration.

See [October 7 verification](../../../docs/verification/2026-10-07-judge-calibration.md) for
tooling checks and the boundary between completed implementation and pending human review.
