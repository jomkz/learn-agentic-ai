# ADR-002: Domain adaptation experiments

Status: **Proposed**

The runnable local baseline is RAG. RAFT and InstructLab are learning extensions; the
repository contains no verified evidence that either improves this baseline. Evaluate them
on a held-out dataset before selecting or promoting a model.

Use the Phase 9 dataset builder to create supervised examples with explicit oracle inclusion
and complete answer targets. Validate each example, separate training from evaluation data,
and record the random seed. Adapter compatibility must be checked against the exact base
model revision; do not assume adapters transfer across base model releases.

Track actual training duration, resource use, general-capability regressions, and retrieval
quality. The evaluation report under `../../evaluation/ragas_comparison.md` describes the
required artifacts. GPU training and model promotion remain optional, unverified exercises.
