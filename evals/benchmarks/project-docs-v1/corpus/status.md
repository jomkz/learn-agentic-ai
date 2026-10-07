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

Unit tests use explicit mocks for optional SDKs. KFP compilation runs with the real SDK when
installed. Full cluster deployment, GPU training, and live-model quality are not implied by
unit-test coverage. The model security workflow evaluates an explicitly supplied endpoint;
a successful scan only covers its selected probes and detector configuration.
