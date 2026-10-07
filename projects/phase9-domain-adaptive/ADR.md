# Domain adaptation decision — proposed experiment

Status: **Proposed; comparative evaluation pending.**

Start with a measured RAG baseline. GraphRAG, RAFT, and InstructLab are optional experiments,
not established quality improvements for this corpus. The previous claim of a completed
50-question evaluation was unsupported by reproducible artifacts and has been withdrawn.

Use `mobility_ai.phase9.capstone` to compare actual outputs on identical held-out questions.
Record corpus and dataset versions, model identifiers, prompts, code revision, per-question
scores, total query latency, and measured cost. Keep training and evaluation sets separate.

Adopt an advanced strategy only if repeated measured improvements justify its indexing,
training, serving, and maintenance costs. Check regressions on unrelated domains as well as
the target domain. No adapter has been promoted or deployment verified in this workspace.
