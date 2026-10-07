# Implementation status and completion criteria

The curriculum is a learning plan. A technology appearing in a phase is not evidence that
its production integration is implemented. Code is under `src/mobility_ai`; phase tests and
supporting material remain under `projects`.

## Core track

Complete Phases 1–3, the bounded tools in Phase 6, and the local capstone first. This covers
provider calls, typed data, retrieval, source attribution, tool boundaries, persistence, and
evaluation. Then add Phase 4 optimizations and Phase 5 orchestration only after recording a
baseline. Phase 7 and the infrastructure/domain-adaptation tracks are optional branches.

| Phase | Current status | Completion evidence | Requirements beyond Python |
|---|---|---|---|
| 1 | Local examples; provider calls optional | Validate configs and pass phase tests | Ollama for model calls; cloud keys optional |
| 2 | Local chain/API demos; simulated search | Pass API/tool tests; show a cited model response | LangChain extra; Ollama for live generation |
| 3 | Chunking/retrieval exercises | Ingest and query persisted documents through the capstone | RAG extra for optional parsing/vector integrations |
| 4 | Exact cache, retrieval patterns, DSPy exercise | Reject unrelated cache hits; run async HyDE; compare actual outputs | Advanced RAG extra; Redis for shared cache |
| 5 | Graph orchestration demo with placeholder research nodes | Demonstrate state transitions; replace nodes and record behavior | Agents extra; live providers for research |
| 6 | Restricted file tools and task API demo | Traversal/symlink tests pass; use an approved file root | POSIX/Linux for descriptor-based file restrictions |
| 7 | Provider portability exercise | Verify each selected backend against its installed SDK/server | LlamaStack server or provider credentials |
| 8 | KFP compilation; training/deployment exercises | Compile the pipeline; separately capture a cluster run | Pipelines extra; GPU/cluster for training and serving |
| 9 | Seeded RAFT data and recorded-run comparison | Check gold targets; compare held-out outputs without score bonuses | Graph/ML extras and services for advanced experiments |
| Capstone | Runnable local workflow | Reopen database, retrieve, cite, record outputs, evaluate | None for lexical mode; Ollama for model-backed mode |

The offline track requires no GPU and only small text fixtures. Model-backed RAM/VRAM needs
are determined by model size, quantization, and context length; measure them on your machine.
GPU training and OpenShift exercises need separately provisioned infrastructure and should
be scheduled after the local completion criteria pass.

## Verification boundaries

Live Ollama CPU ingestion, cited answering, abstention, idempotent updates, and failed-update
preservation were verified on 2026-09-23. See the
[recorded documentation benchmark](../evals/benchmarks/project-docs-v1/results/2026-09-23/README.md)
for the tested models, quality limitations, and reproducible artifacts.

Unit tests use explicit mocks for optional SDKs. KFP compilation runs with the real SDK when
installed. Full cluster deployment, GPU training, and live-model quality are not implied by
unit-test coverage. The model security workflow evaluates an explicitly supplied endpoint;
a successful scan only covers its selected probes and detector configuration.

The [October 7 closeout verification](verification/2026-10-07.md) records a fresh package
installation, regression checks, isolated phase installs, and offline notebook execution.

## Follow-up quality work

The local workflow and September 23 benchmark form the baseline for the next iteration.
Prioritize the following work using the recorded failure cases:

1. **Structured responses — complete:** answers, citations, and abstention now have explicit
   fields. Missing citations fail validation; alternate abstention wording renders consistently.
   [October 7 verification](verification/2026-10-07-structured-responses.md) records 433 passing
   tests and eight valid live responses. One live answer also cited an irrelevant passage;
   citation support remains priority 3.
2. **Chunk boundaries — next:** compare sentence-aware chunking or adjacent-chunk expansion against
   the truncated cache-namespace example, while retaining source provenance.
3. **Citation support:** distinguish an existing source identifier from a passage that supports
   the claim. Include the wrong-passage, extra-passage, and unsupported-reasoning examples in
   development checks.
4. **Judge calibration:** use a larger, manually reviewed set containing supported claims,
   unsupported claims, non-answers, and incorrect citations. Report judge disagreements;
   the current judge is not a correctness gate.

Develop against the inspected cases, then evaluate on a new held-out question set. Preserve
the frozen benchmark and its original results. GPU training, cluster execution, and a live
security scan require separately configured infrastructure and remain optional follow-up work.
