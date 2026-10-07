# Judge-calibration tooling verification — 2026-10-07

Follow-up priority 4 now has a reproducible, unlabelled independent-review packet and a
comparison command. **Human review is deferred.** Calibration is not complete: no human
labels, new live judge run, or agreement report exists for this 32-case set. The existing
citation judge and capstone acceptance behavior are unchanged.

## Implemented workflow

The [packet](../../evals/calibration/citation-support/suite/packet/review.html) contains 32 cases
and 38 citations. Eleven cases retain the earlier inspected examples; 21 authored controls
extend coverage of quantities/units, negation, qualification, list scope, non-answers,
conflicting evidence, distributed/uncited evidence, extra citations, and unsupported reasoning.
It is development calibration material, not held-out or independently labeled data.

The local browser form contains no suggested labels or judge predictions. It supports draft
save/load and completed JSON export, with no automatic persistence or network requests.
Only cited evidence is shown; filenames, reference answers, and uncited text are excluded.
Markup in recorded inputs is displayed as text. Case IDs and input hashes bind the exported
review to the precise dataset. The independent-review checkbox is a reviewer attestation.

The comparison requires complete independent labels and a completed audit on the same input.
It rejects duplicate/missing cases and citations, altered answers/passages, invalid excerpts,
and saved judgments that differ from their raw judge responses. Judge failures remain in
overall agreement denominators; valid-only rates are reported separately. Answer support
and individual citation contribution have separate confusion counts and disagreement lists.
All explanations remain available, with rationale review explicitly pending even on matches.

## Validation

- All 558 tests passed with 93.33% branch-inclusive coverage, above the 90% gate.
- 55 targeted tests cover review completeness, provenance, raw-response consistency, error
  accounting, null conditional rates, separate citation disagreement, input eligibility,
  prompt isolation, markup escaping, CLI behavior, and archived suite hashes.
- Ruff lint/format passed for 116 Python files; mypy passed for 55 source files.
- Fourteen headless Chrome checks exercised actual browser controls: initial blank state,
  incomplete-export rejection, draft saving, progress, complete export, draft restoration,
  ID-based import despite reordering, wrong-input/duplicate/invalid-label rejection, and
  preservation of current work when import fails. Only synthetic test labels were used;
  none became calibration labels or results.
- The wheel was built and installed outside the source tree. Its packaged HTML template
  generated a byte-identical review packet from the archived records.
- All five suite artifacts reproduced byte-for-byte from the preparation script.
- The eight frozen corpus files and all 54 earlier benchmark/development artifacts retain
  their original hashes.

## Resumption criteria

The [workflow instructions](../../evals/calibration/citation-support/README.md) record exact
commands to resume. First complete and freeze independent labels; then run the unchanged
`citation-support-v2` judge, compare every case, and review rationales including label matches.
Preserve errors and disagreements, resolve ambiguous labels with further review, and retain
model/settings/source hashes. No model-generated substitute labels or synthetic accuracy
claims are used while human review is pending. A later held-out answer evaluation remains
separate from this development calibration work.
