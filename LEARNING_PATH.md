# Learning Path: Agentic AI & MLOps on OpenShift AI

## Context

This learning path is for a senior software engineer/architect at Red Hat who needs to become proficient in designing, building, and leading the implementation of production-grade Agentic AI applications. The target stack is Python-first, open-source-first, and Red Hat/OpenShift-aligned.

**Goal activities:**
1. Architect, design, document, and develop Python apps using LangChain, LangGraph, and LlamaStack
2. Architect and lead implementation of Agentic AI applications
3. Architect scalable open-source ML solutions with distributed computing on OpenShift AI
4. Architect and design features using RAG, RAFT, GraphRAG, InstructLab, and their pipelines
5. Develop and optimize RAG pipelines

---

## Learning Path Overview

**Total duration: ~41 weeks (~10 months full-time / 12+ months at 20 hrs/week)**

| Phase | Topic | Duration | Cumulative |
|-------|-------|----------|------------|
| 1 | Foundations + Python Modernization + Local Dev | 3 weeks | 3 weeks |
| 2 | LangChain + LCEL + LangSmith + Streaming + Testing | 5 weeks | 8 weeks |
| 3 | RAG Fundamentals + Document Parsing + Vector Stores | 4 weeks | 12 weeks |
| 4 | Advanced RAG + Caching + Cost Optimization | 3 weeks | 15 weeks |
| 5 | LangGraph + Multi-Agent Workflows | 4 weeks | 19 weeks |
| 6 | Agentic Patterns + MCP + Guardrails | 3 weeks | 22 weeks |
| 7 | LlamaStack | 3 weeks | 25 weeks |
| 8 | HuggingFace + LoRA/QLoRA + OpenShift AI + vLLM + KFP + Ray + MLOps | 8 weeks | 33 weeks |
| 9 | GraphRAG + RAFT + InstructLab + Multi-Modal | 6 weeks | 39 weeks |
| Final | End-to-End Capstone Platform | 2 weeks | 41 weeks |

**Dependency map:**
```
Phase 1 (Foundations + Local Dev)
    ├── Phase 2 (LangChain + LCEL + Testing)
    │       ├── Phase 3 (RAG Fundamentals)
    │       │       └── Phase 4 (Advanced RAG + Caching)
    │       │               └── Phase 9 (GraphRAG/RAFT/InstructLab)
    │       └── Phase 5 (LangGraph)
    │               └── Phase 6 (Agentic Patterns + MCP + Guardrails)
    └── Phase 7 (LlamaStack) [can start after Phase 2]
Phase 8 (HuggingFace + OpenShift AI) [can start in parallel with Phase 5; needs Phase 3]
Phase 9 (GraphRAG/RAFT/InstructLab)  [needs Phases 4, 8]
Final Capstone                        [needs all phases]
```

---

## Phase 1: AI/ML Foundations, Python Modernization, and Local Dev Environment
**Duration: 3 weeks**

### Objectives
- Understand how LLMs work: tokens, attention, context windows, temperature, sampling
- Master modern Python patterns used across all AI frameworks: async/await, type hints, Pydantic v2
- Set up a productive local development environment with offline model serving via Ollama
- Understand model formats and what drives the choice between them

### Key Concepts

#### LLM Fundamentals
- Tokenization, embeddings, attention mechanism (conceptual, not mathematical)
- Context windows, temperature, top-p, top-k sampling
- Prompt engineering: system prompts, few-shot examples, chain-of-thought prompting
- LLM API patterns: completion vs. chat, streaming vs. blocking, token limits

#### Modern Python for AI
- Pydantic v2: `BaseModel`, field validators, `model_config`, discriminated unions, `model_json_schema()`
- Python `async`/`await`: event loops, `asyncio.gather`, `TaskGroup`, async generators, `async for`
- Python typing: generics, `TypeVar`, `Protocol`, `Annotated`, `Literal`, `TypedDict`
- `uv`: fast Python package manager — workspace management, lockfiles, tool installs

#### Local Model Development with Ollama
- Why local models matter: no API costs during development, data privacy, offline capability
- Ollama architecture: model library, REST API, OpenAI-compatible endpoint
- Running models locally: `ollama pull`, `ollama run`, `ollama serve`
- Interacting via Python: `ollama` Python library and OpenAI SDK pointed at `http://localhost:11434/v1`
- Choosing local models for dev: Llama 3.2 (3B for fast iteration), Llama 3.1 8B (quality)
- Model file formats overview:
  - **GGUF**: CPU/GPU quantized format used by Ollama/llama.cpp; most accessible
  - **safetensors**: HuggingFace's secure, fast-loading model format; used in training
  - **GPTQ / AWQ**: GPU-quantized formats for vLLM deployment
  - Quantization tradeoffs: 4-bit vs 8-bit vs full precision — quality vs memory vs speed

### Resources
- Andrej Karpathy — "Intro to Large Language Models" (YouTube, 1hr): https://youtu.be/zjkBMFhNj_g
- Andrej Karpathy — "Let's build GPT from scratch" (YouTube): https://youtu.be/kCc8FmEb1nY
- "The Illustrated Transformer" by Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- Anthropic Prompt Engineering Guide: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
- FastAPI docs (excellent async + Pydantic patterns): https://fastapi.tiangolo.com/
- uv package manager: https://docs.astral.sh/uv/
- Ollama docs: https://ollama.com/docs
- Ollama GitHub: https://github.com/ollama/ollama
- Ollama Python library: https://github.com/ollama/ollama-python

### Hands-on Projects
1. **Dev environment setup**: Install `uv`, Ollama, pull Llama 3.2 3B and Llama 3.1 8B; verify local inference via Python; set up a Jupyter environment
2. **Pydantic v2 model**: Build an "AI Agent configuration" model — nested models, discriminated unions for provider selection (`openai` vs `ollama` vs `anthropic`), field validators
3. **Async LLM client**: Async Python script calling both Ollama (local) and an LLM API concurrently with multiple prompts; measure and compare latency
4. **Prompt engineering iteration**: Take a vague task and iterate on system prompt + few-shot examples; test against both local Llama and cloud model to understand quality gaps

---

## Phase 2: LangChain + LCEL + LangSmith + Streaming + Testing
**Duration: 5 weeks**

### Objectives
- Build LLM applications using LCEL pipe composition
- Implement agents with tools, memory, and multi-turn conversations
- Use LangSmith for tracing, debugging, and systematic evaluation
- Produce streaming AI responses and structure reliable tool outputs
- Write meaningful tests for LLM-powered applications

### Key Concepts

#### LCEL and Chains
- LCEL: pipe operator `|`, `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`, `RunnableBranch`
- Output parsers: `StrOutputParser`, `PydanticOutputParser`, `JsonOutputParser`
- Structured output: `model.with_structured_output(MySchema)` — the reliable way to get typed data from any LLM
- Document loaders, text splitters, vector store integration (Chroma/FAISS for this phase)
- Chat history management: `MessagesPlaceholder`, `RunnableWithMessageHistory`
- Prompt management: LangSmith Hub — storing, versioning, and pulling prompts with `hub.pull("owner/prompt-name")`

#### Agents and Tools
- `create_tool_calling_agent` + `AgentExecutor` — the standard agent pattern
- `@tool` decorator, `BaseTool` subclass, `StructuredTool.from_function()`
- Tool schemas: how JSON Schema drives what the model knows about a tool's inputs
- Memory patterns: `ConversationBufferMemory`, `ConversationSummaryMemory`, `ConversationTokenBufferMemory`

#### Streaming
- `chain.stream()` and `chain.astream()` — synchronous and async streaming
- `chain.astream_events()` — fine-grained event stream: `on_chat_model_stream`, `on_tool_start`, `on_tool_end`
- Server-Sent Events (SSE) pattern: streaming LLM responses from a FastAPI endpoint to a browser client
- Backpressure and buffering considerations for streaming in production

#### LangSmith
- Project setup, environment variables, automatic tracing with zero code changes
- Manual tracing: `@traceable` decorator, `with_` context manager
- Datasets and examples: building ground-truth QA pairs for evaluation
- Evaluators: `LLMAsJudge`, `ExactMatch`, custom Python evaluators
- LangSmith Hub for prompt version control

#### Testing LLM Applications
- The challenge: LLM outputs are non-deterministic; tests need different strategies than unit tests
- **Deterministic tests**: test that the right tools were called, that the output is parseable, that schemas validate
- **Regression tests**: golden dataset comparison — does the new prompt produce answers at least as good?
- **Semantic assertion tests**: use an embedding similarity threshold or LLM judge to assert that output "means" the right thing
- `pytest` + `pytest-asyncio` for async LangChain code
- Mocking LLM calls: `langchain_core.runnables.testing`, `FakeChatModel`
- `DeepEval` framework for structured LLM test assertions: https://docs.confident-ai.com/
- LangSmith evaluation runs as a CI step (run eval suite on PR, compare scores to baseline)

### Resources
- LangChain Python docs: https://python.langchain.com/docs/introduction/
- LCEL guide: https://python.langchain.com/docs/concepts/lcel/
- LangSmith docs: https://docs.smith.langchain.com/
- LangChain GitHub: https://github.com/langchain-ai/langchain
- DeepLearning.AI "LangChain for LLM Application Development" (free, ~3hr): https://learn.deeplearning.ai/courses/langchain
- DeepLearning.AI "LangChain: Chat with Your Data" (free): https://learn.deeplearning.ai/courses/langchain-chat-with-your-data
- Sam Witteveen YouTube (@samwitteveenai): practical LangChain deep dives
- ReAct paper: https://arxiv.org/abs/2210.03629
- DeepEval docs: https://docs.confident-ai.com/
- FastAPI streaming guide: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse

### Hands-on Projects
1. **Document Q&A Chain**: Load PDFs, chunk, embed into Chroma, LCEL retrieval chain with custom prompt, traced in LangSmith; prompt stored and versioned in LangSmith Hub
2. **Streaming Chat API**: FastAPI endpoint that streams LLM responses via SSE using `astream_events`; test from a simple HTML client
3. **Structured Output Agent**: Agent with 3+ tools that always returns a validated Pydantic model; no raw string parsing allowed
4. **LLM Test Suite**: `pytest` suite with deterministic tests, golden dataset regression, and `DeepEval` semantic assertions for the Document Q&A chain

### Capstone
**"Research Assistant Agent"** — Tool-calling agent that searches multiple sources, synthesizes information, and streams a structured report to a FastAPI SSE endpoint. Fully traced in LangSmith. Test suite covers schema validation, tool invocation, and semantic quality regression.

---

## Phase 3: RAG Fundamentals — Document Parsing, Embeddings, Vector Stores, Retrieval
**Duration: 4 weeks**

### Objectives
- Parse complex real-world documents (PDFs with tables, images, mixed formats)
- Understand embeddings, similarity metrics, and chunking tradeoffs deeply
- Work with production-grade vector databases: pgvector, Milvus, Qdrant, Weaviate
- Understand where LlamaIndex fits alongside LangChain for RAG workloads
- Build a systematic RAGAS evaluation pipeline reusable across all future phases

### Key Concepts

#### Document Parsing (the underrated first step)
- Why document parsing matters: garbage in, garbage out — poor parsing kills RAG quality
- **Docling** (Red Hat open-source): intelligent PDF/DOCX/HTML parsing preserving tables, headings, lists, and image captions; outputs structured JSON or Markdown
  - `DoclingLoader` integrates directly with LangChain
  - Handles complex PDFs that PyPDF/pdfplumber fail on (multi-column, scanned, tables)
- `Unstructured.io`: general-purpose document parsing for diverse formats
- `pymupdf4llm`: fast, lightweight PDF-to-Markdown conversion
- When to use each: Docling for structure-critical enterprise docs; pymupdf4llm for speed; Unstructured for format variety
- Image extraction and multi-modal document processing (brief intro — deep coverage in Phase 9)

#### Embeddings
- Embedding models: dense vs sparse vectors
  - Dense: BGE (`BAAI/bge-large-en-v1.5`), E5 (`intfloat/e5-large-v2`), Sentence Transformers (`all-mpnet-base-v2`), OpenAI `text-embedding-3-small/large`
  - Sparse: BM25 (term frequency), SPLADE
- Similarity metrics: cosine (normalized), dot product (unnormalized), L2 — when each applies
- Running embedding models locally with Ollama: `ollama pull nomic-embed-text`

#### Chunking Strategies
- Fixed-size with overlap: simple, predictable, loses structure
- Recursive character: LangChain's default, respects paragraph/sentence boundaries
- Semantic chunking: split on embedding similarity drops between sentences
- Document-structure-aware: chunk by heading, section, or element type (Docling enables this)
- Chunk overlap and its effect on retrieval boundary cases
- Metadata enrichment: attach source, section, page number to every chunk

#### Vector Stores (Comparison)
| Store | Best for | Notes |
|-------|----------|-------|
| **pgvector** | Production default; SQL co-location | Runs in existing PostgreSQL; HNSW indexing |
| **Qdrant** | High-scale production; rich filtering | Rust-based; excellent payload filtering; cloud or self-hosted |
| **Milvus** | Distributed, very large scale | Kubernetes-native; more ops overhead |
| **Weaviate** | Schema-first; built-in vectorization | GraphQL API; good for structured data |
| **Chroma** | Local dev and prototyping | Zero-ops; not for production |
| **FAISS** | Research and offline batch search | In-memory/file; no serving layer |

- Hybrid search: dense (semantic) + sparse (BM25) combined with Reciprocal Rank Fusion (RRF)
- Metadata filtering: pre-filter by source, date, category before vector similarity ranking
- Vector store internals: HNSW (graph-based, approximate), IVF (inverted file, cluster-based), flat (exact, slow)

#### LlamaIndex vs. LangChain for RAG
LlamaIndex is the other major RAG framework alongside LangChain — know both:
- **LlamaIndex strengths**: richer built-in index types (VectorStoreIndex, KnowledgeGraphIndex, SummaryIndex), deeper document hierarchy modeling, excellent for complex structured retrieval
- **LangChain strengths**: broader ecosystem (agents, tools, chains), better for multi-step pipelines beyond retrieval
- **In practice**: many production systems use LangChain for orchestration + LlamaIndex concepts for retrieval design, or pick one and stick with it
- LlamaIndex `VectorStoreIndex`, `QueryEngine`, `RetrieverQueryEngine`
- LlamaIndex docs: https://docs.llamaindex.ai/en/stable/
- LlamaIndex GitHub: https://github.com/run-llama/llama_index

#### Evaluation
- RAGAS metrics: faithfulness, answer relevancy, context precision, context recall, answer correctness
- Building a reusable evaluation harness: dataset of (question, ground-truth-answer, retrieved-context) triples
- Automated evaluation with LLM judges vs. human evaluation — cost/quality tradeoffs

### Resources
- Docling GitHub: https://github.com/DS4SD/docling
- Docling docs: https://ds4sd.github.io/docling/
- Unstructured.io docs: https://docs.unstructured.io/
- pgvector GitHub: https://github.com/pgvector/pgvector
- Qdrant docs: https://qdrant.tech/documentation/ | GitHub: https://github.com/qdrant/qdrant
- Milvus docs: https://milvus.io/docs
- Weaviate docs: https://weaviate.io/developers/weaviate
- Chroma docs: https://docs.trychroma.com/
- LlamaIndex docs: https://docs.llamaindex.ai/en/stable/
- RAGAS paper: https://arxiv.org/abs/2309.15217 | RAGAS docs: https://docs.ragas.io/
- DeepLearning.AI "Building and Evaluating Advanced RAG" (free): https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag
- Greg Kamradt's chunking tutorials: https://github.com/FullStackRetrieval-com/RetrievalTutorials
- Sentence-BERT paper: https://arxiv.org/abs/1908.10084

### Hands-on Projects
1. **Document Parsing Comparison**: Take a complex multi-column PDF with tables. Parse with Docling, pymupdf4llm, and PyPDF. Measure: table extraction quality, heading preservation, chunk coherence
2. **Embedding Comparison**: Same corpus, 3 embedding models (BGE, E5, OpenAI `text-embedding-3-small`); measure retrieval quality on a hand-curated QA set with RAGAS
3. **Chunking Experiment**: Same Docling-parsed document, 4 chunking strategies; measure RAGAS score impact; identify which strategy wins for which document type
4. **pgvector + Qdrant Production Setup**: Same dataset loaded into both; compare query latency, metadata filtering capability, and operational complexity
5. **RAGAS Evaluation Harness**: Reusable pipeline in `evals/` that takes any (retriever, dataset) pair and produces a RAGAS scorecard — used in all future phases

### Capstone
**"Technical Documentation Search"** — Production RAG over Kubernetes docs. Docling for parsing. pgvector + hybrid search. RAGAS evaluation. FastAPI service with SSE streaming (from Phase 2). Deployed locally with Docker Compose / Podman Compose.

---

## Phase 4: Advanced RAG Techniques + Semantic Caching + Cost Optimization
**Duration: 3 weeks**

### Objectives
- Implement retrieval strategies that measurably outperform naive RAG
- Add semantic caching to reduce latency and LLM cost
- Build cost-optimization patterns essential for production AI architectures
- Establish evaluation-driven improvement loops

### Key Concepts

#### Advanced Retrieval Techniques
- **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer, embed it for retrieval — improves semantic match for vague queries
- **Multi-query retrieval**: generate N query reformulations, retrieve for each, fuse with Reciprocal Rank Fusion (RAG-Fusion)
- **Contextual compression**: compress retrieved chunks down to only the relevant excerpt before passing to LLM
- **Step-back prompting**: abstract the question before retrieving (e.g., "what is quantum entanglement" → "what is quantum mechanics")
- **Parent-child chunking**: small chunks for retrieval precision, return surrounding large chunk for context
- **Sentence window retrieval**: retrieve at sentence level, return N surrounding sentences
- **Cross-encoder reranking**: BGE-reranker, Cohere Rerank API, FlashRank (local, fast) — rescores top-k candidates
- **Self-RAG**: model emits special tokens to decide when to retrieve and critiques its own output
- **CRAG** (Corrective RAG): assess retrieval quality; if poor, fall back to web search

#### Semantic Caching
- Why cache: LLM inference is expensive and slow; many real-world queries are semantically similar
- **Redis Semantic Cache** (`langchain_community.cache.RedisSemanticCache`): embed the query, check vector similarity in Redis; return cached response if above threshold
- Cache hit rate vs. staleness tradeoffs: TTL configuration, invalidation strategies
- Cache warming: pre-populate with known frequent queries
- When NOT to cache: highly personalized responses, real-time data queries, low-repetition workloads

#### LLM Cost Optimization Patterns
- **Model tiering**: use a cheap/fast model (Llama 3.2 3B, GPT-4o-mini, Haiku) for routing, classification, and summarization; reserve expensive models for final synthesis
- **Prompt caching**: Anthropic and OpenAI both support prompt prefix caching — cache the system prompt + documents prefix for repeated calls; can reduce cost by 80-90% for RAG workloads
- **Batching**: group multiple embedding requests; use async for concurrent LLM calls
- **Token budgeting**: count tokens before calling LLM (`tiktoken`, `transformers` tokenizer); truncate context dynamically to stay under budget
- **Streaming for perceived performance**: even if total time is the same, streaming starts showing results immediately
- Context window management:
  - Stuffing: stuff all retrieved context into one call (simple but hits limits)
  - Map-reduce: process chunks independently, then reduce
  - Refine: iteratively refine answer over each chunk
  - Summarize-then-query: compress long documents before adding to context

### Resources
- HyDE paper: https://arxiv.org/abs/2212.10496
- Self-RAG paper: https://arxiv.org/abs/2310.11511
- CRAG paper: https://arxiv.org/abs/2401.15884
- RAG Survey (comprehensive): https://arxiv.org/abs/2312.10997
- Lance Martin's "RAG from Scratch": https://github.com/langchain-ai/rag-from-scratch
- DeepLearning.AI "Building and Evaluating Advanced RAG": https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag
- Redis semantic cache docs: https://python.langchain.com/docs/integrations/llm_caching/
- Anthropic prompt caching docs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- tiktoken (token counting): https://github.com/openai/tiktoken

### Hands-on Projects
1. **Technique Benchmarking**: HyDE vs Multi-Query vs Reranking on the Phase 3 RAGAS harness — compare quality, latency, and per-query cost
2. **Reranker Integration**: BGE-reranker (local) + Cohere Rerank (API); measure precision improvement at top-3 and top-5
3. **Semantic Cache Implementation**: Add Redis semantic cache to the Phase 3 documentation search; measure cache hit rate on a realistic query workload; calculate cost savings
4. **Cost-Optimized Architecture**: Refactor the Phase 2 Research Agent to use model tiering — cheap model for tool routing, expensive model only for final answer synthesis; measure cost reduction

---

## Phase 5: LangGraph — Stateful Agents and Multi-Agent Workflows
**Duration: 4 weeks**

### Objectives
- Build complex, stateful agent workflows with checkpointing and fault tolerance
- Implement human-in-the-loop approval and review flows with streaming
- Design production multi-agent architectures for real business problems

### Key Concepts
- `StateGraph`: nodes, edges, `END`, conditional edges with router functions
- State management: `TypedDict` for simple state, Pydantic `BaseModel` for validated state
- Compiled graphs: `graph.compile()`, `graph.invoke()`, `graph.stream()`, `graph.astream_events()`
- Checkpointing: `MemorySaver` (dev), `SqliteSaver` (persistent dev), `PostgresSaver` (production)
- Thread IDs and conversation isolation: how checkpoints map to user sessions
- Interrupts: `interrupt_before`, `interrupt_after` — pause graph for human input
- Human-in-the-loop flows: `Command(resume=value)` to continue after approval
- Subgraphs and graph composition: building modular agent systems
- Streaming modes: `values` (full state per step), `updates` (deltas only), `debug` (everything)
- Multi-agent patterns:
  - **Supervisor**: a router node calls an LLM to decide which specialized agent to hand off to
  - **Hierarchical**: supervisors of supervisors — for very complex domain decomposition
  - **Swarm / handoffs**: agents pass control directly using `Command(goto="agent_name")`
- Long-term memory store: `InMemoryStore` → `PostgresStore` for facts that persist across threads
- Time travel: replaying or branching from a previous checkpoint state (excellent for debugging)

### Resources
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangGraph tutorials: https://langchain-ai.github.io/langgraph/tutorials/
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- DeepLearning.AI "AI Agents in LangGraph" (free): https://learn.deeplearning.ai/courses/ai-agents-in-langgraph
- Plan-and-Solve paper: https://arxiv.org/abs/2305.04091
- Autonomous Agents survey: https://arxiv.org/abs/2308.11432

### Hands-on Projects
1. **Stateful Research Agent**: Rebuild Phase 2 Research Agent in LangGraph — persistent checkpointing, human approval node before web searches, streaming via `astream_events`, time-travel debugging
2. **Multi-Agent Code Review Pipeline**: Supervisor routes code review tasks to specialized subagents (security, performance, style); each streams findings; supervisor synthesizes a final report
3. **Human-in-the-Loop Content Workflow**: Agent drafts content → human approves or edits → agent revises and publishes; demonstrates full interrupt/resume/state persistence cycle

### Capstone
**"Autonomous Research Pipeline"** — Multi-agent system: supervisor orchestrates a research agent (web + RAG from Phase 3), a data analysis agent (Python code interpreter), and a writing agent. Fully checkpointed with PostgresSaver. Streams results via the Phase 2 SSE endpoint. LangSmith evaluation comparing output quality to Phase 2 single-agent baseline.

---

## Phase 6: Agentic Patterns, MCP, Guardrails, and Context Management
**Duration: 3 weeks**

### Objectives
- Master production agentic AI design patterns
- Build and consume MCP (Model Context Protocol) servers
- Add output guardrails and safety rails to production agents
- Implement reliable context window management strategies

### Key Concepts

#### Agentic Design Patterns
- **ReAct** (Reason + Act): interleave reasoning traces with action calls — the default LangChain/LangGraph agent mode
- **Plan-and-Execute**: explicitly generate a plan, then execute each step — better for complex multi-step tasks
- **Reflexion**: self-critique loop — agent evaluates its own output and retries if quality is insufficient
- **Tree of Thoughts**: explore multiple reasoning paths in parallel, select the best
- **LATS** (Language Agent Tree Search): MCTS-style tree search over action sequences

#### Model Context Protocol (MCP)
- MCP motivation: standardize tool exposure so any LLM client can use any tool server
- Architecture: MCP Host (Claude Desktop, LangChain, LlamaStack) ↔ MCP Client ↔ MCP Server
- Transports: stdio (local process, most common), SSE (HTTP, for remote servers)
- MCP primitives: **Tools** (callable functions), **Resources** (readable data), **Prompts** (reusable templates)
- Building MCP servers in Python with the `mcp` SDK: `@server.tool()`, `@server.resource()` decorators
- Consuming MCP servers: LangChain `MultiServerMCPClient`, LlamaStack tool groups
- Community MCP servers: filesystem, PostgreSQL, GitHub, web fetch, Docker

#### Guardrails and Safety
- **Prompt injection defense**: system prompt hardening, instruction hierarchy, input sanitization
- **Output validation**: always parse/validate agent outputs against a schema before acting on them
- **NeMo Guardrails** (NVIDIA): declarative `.co` rail files for topic control, fact-checking, moderation; integrates with LangChain: https://github.com/NVIDIA/NeMo-Guardrails
- **Guardrails AI**: Python-first; validators run on output (PII detection, profanity, JSON schema, semantic similarity): https://www.guardrailsai.com/docs
- **LlamaGuard** (Meta): LLM-based content classifier for safety; runs as a shield in LlamaStack
- Output guardrail pattern: `agent_output → guardrail_check → pass/block/rephrase`

#### Context Window Management
- Token counting before LLM calls: `tiktoken.encoding_for_model()`, HuggingFace `AutoTokenizer`
- Dynamic context truncation: prioritize recent messages and retrieved context; drop oldest
- Summarization for long conversations: `ConversationSummaryMemory` or custom graph node
- Long-context models and their tradeoffs: Llama 3.1 (128K), Gemini (1M) — longer context ≠ better attention
- "Lost in the middle" problem: models attend poorly to content in the middle of long contexts; critical content should be at the start or end
- Agent reliability patterns: timeout + retry with exponential backoff, fallback chains, cost/token budget enforcement per agent step

### Resources
- MCP official docs: https://modelcontextprotocol.io/docs
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP GitHub (spec + community servers): https://github.com/modelcontextprotocol
- NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails | Docs: https://docs.nvidia.com/nemo/guardrails
- Guardrails AI: https://www.guardrailsai.com/docs | GitHub: https://github.com/guardrails-ai/guardrails
- LlamaGuard paper: https://arxiv.org/abs/2312.06674
- ReAct paper: https://arxiv.org/abs/2210.03629
- Reflexion paper: https://arxiv.org/abs/2303.11366
- "Building Effective Agents" — Anthropic blog: https://www.anthropic.com/research/building-effective-agents
- "Lost in the Middle" paper: https://arxiv.org/abs/2307.03172

### Hands-on Projects
1. **MCP Server**: Build a Python MCP server exposing 3 tools (filesystem access, SQL DB query, HTTP API wrapper). Test with Claude Desktop and programmatically via Python MCP client
2. **MCP + LangGraph**: Integrate the MCP server tools into a LangGraph agent; demonstrate they're interchangeable with native LangChain tools
3. **Guardrailed Agent**: Add NeMo Guardrails rail file + Guardrails AI output validators to the Phase 5 Research Pipeline; test that prompt injection attempts are blocked and PII is redacted from outputs
4. **Context Budget Manager**: Utility class that counts tokens, dynamically trims chat history, and enforces a per-call token budget; integrate into the Phase 5 agent

---

## Phase 7: LlamaStack
**Duration: 3 weeks**

### Objectives
- Understand LlamaStack's provider abstraction and distribution model
- Build applications and agentic loops using LlamaStack's native APIs
- Understand how LlamaStack complements LangChain/LangGraph in a production architecture

### Key Concepts
- LlamaStack motivation: standardize the AI application stack the same way the Linux kernel standardizes OS
- Architecture layers: distributions → providers → APIs
- Distribution types: Ollama (local), Together AI, Fireworks, TGI, custom (any OpenAI-compatible server)
- Core APIs:
  - **Inference**: chat completions, streaming, tool calling
  - **Safety**: shields, LlamaGuard integration
  - **Memory**: memory banks for agent state (in-memory, vector, key-value)
  - **Agents**: agentic loop with tool groups, multi-turn sessions
  - **DatasetIO**: dataset management for evals and training
  - **Scoring**: evaluation and benchmarking
- LlamaStack Agents: built-in agentic loop, tool groups (web search, code interpreter, memory), session management
- Safety providers: `llama_guard` shield, configuring content filtering in a distribution YAML
- LlamaStack Python client: async-first API design
- Running LlamaStack with Ollama: local development with no cloud dependency
- LlamaStack on OpenShift: deploying a LlamaStack distribution as a container (Phase 8 connection)

### Key distinction from LangChain/LangGraph
LlamaStack is an **infrastructure layer and API standard** — it defines *how* you serve models and attach tools. LangChain/LangGraph is an **orchestration layer** — it defines *how* you chain calls and build graphs. They compose: LangGraph as the brain, LlamaStack as the model serving and safety substrate.

### Resources
- LlamaStack docs: https://llama-stack.readthedocs.io/en/latest/
- LlamaStack GitHub: https://github.com/meta-llama/llama-stack
- LlamaStack Python client: https://github.com/meta-llama/llama-stack-client-python
- LlamaStack Apps examples: https://github.com/meta-llama/llama-stack-apps
- LlamaStack distributions reference: https://llama-stack.readthedocs.io/en/latest/distributions/

### Hands-on Projects
1. **LlamaStack + Ollama Distribution**: Configure and run a LlamaStack server with Ollama provider; write a Q&A application using the LlamaStack Python client; compare ergonomics to the same app with LangChain
2. **LlamaStack Agent with Safety**: Build a LlamaStack native agent with LlamaGuard shield enabled; verify the shield blocks unsafe content; test tool calling with web search tool group
3. **LlamaStack RAG**: Document retrieval using LlamaStack's memory bank API; store 50 docs, query, return results to an agent
4. **LangGraph + LlamaStack Integration**: LangGraph graph where each LLM node calls the LlamaStack inference API; demonstrates using LangGraph orchestration with LlamaStack serving

### Capstone
**"Portable AI Application"** — Application that switches between a local LlamaStack/Ollama distribution and a cloud provider (via LangChain), controlled by environment configuration. LlamaStack handles inference + safety shields. LangGraph handles orchestration. LangSmith traces the full graph.

---

## Phase 8: HuggingFace Ecosystem, LoRA/QLoRA, OpenShift AI, vLLM, KubeFlow Pipelines, Ray, and MLOps
**Duration: 8 weeks**

### Objectives
- Master the HuggingFace ecosystem as the foundation for all fine-tuning work
- Understand and apply LoRA/QLoRA for parameter-efficient fine-tuning
- Deploy and serve LLM models at scale with vLLM on OpenShift AI
- Build ML data and training pipelines with KubeFlow Pipelines v2
- Use Ray for distributed data processing and model serving
- Implement MLOps: model registry, monitoring, A/B testing
- Work with Podman and container images for ML workloads (Red Hat tooling)

### Key Concepts

#### HuggingFace Ecosystem (weeks 1-2 of this phase)
This is the foundation for all fine-tuning work in Phase 9. Learn it here before the complexity of RAFT and InstructLab.

- **`transformers` library**: `AutoModelForCausalLM`, `AutoTokenizer`, `pipeline()`, `generate()` with sampling parameters
- **`datasets` library**: loading from Hub, local files, streaming large datasets; `map()` for preprocessing; `DatasetDict` structure
- **HuggingFace Hub**: model cards, model repositories, `push_to_hub()`, `hf_hub_download()`, private repos
- **`tokenizers`**: fast tokenizers, batch encoding, special tokens, padding/truncation strategies
- **`accelerate`**: device-agnostic training (`Accelerator`), mixed precision (fp16/bf16), gradient accumulation
- **`PEFT` (Parameter-Efficient Fine-Tuning)**: LoRA, QLoRA, adapter-based methods — the key library for Phase 9
- **`trl` (Transformer Reinforcement Learning)**: `SFTTrainer` for supervised fine-tuning, `DPOTrainer` for preference alignment; key tool for RAFT
- **`evaluate`**: standardized metrics (BLEU, ROUGE, accuracy, perplexity)
- Model formats: `safetensors` (HF default), converting to GGUF for Ollama, exporting to ONNX

#### LoRA and QLoRA — Parameter-Efficient Fine-Tuning (weeks 2-3 of this phase)
Before doing RAFT in Phase 9, understand the fine-tuning fundamentals:

- **Why fine-tuning**: when RAG is not enough — domain-specific style, format, terminology, or behavior that can't be prompted
- **Full fine-tuning**: update all weights — expensive, requires many GPUs, catastrophic forgetting risk
- **LoRA** (Low-Rank Adaptation): freeze original weights, train small rank-decomposed adapter matrices (A and B); `r=8` or `r=16` are common ranks
  - Key parameters: `r` (rank), `lora_alpha` (scaling), `target_modules` (which layers), `lora_dropout`
  - LoRA layers: typically applied to `q_proj`, `v_proj`, `k_proj`, `o_proj` in attention layers
  - At inference: adapters can be merged into base weights or kept separate (supports multiple adapters)
- **QLoRA**: quantize the base model to 4-bit (NF4) with `bitsandbytes`; train LoRA adapters in higher precision on top; dramatically reduces GPU memory (7B model fits on a 12GB GPU)
  - `BitsAndBytesConfig`: `load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16`
  - `prepare_model_for_kbit_training()` from PEFT
- **Training pipeline with PEFT + TRL**:
  1. Load base model with quantization config
  2. Apply `get_peft_model(model, lora_config)` to wrap with LoRA adapters
  3. Prepare training data in chat format
  4. Train with `SFTTrainer` — handles padding, masking, packing
  5. Save adapter weights with `model.save_pretrained()`
  6. Optionally merge adapters: `model.merge_and_unload()` → save full model
- **Evaluation during training**: `eval_dataset`, `compute_metrics`, perplexity tracking
- LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" https://arxiv.org/abs/2106.09685
- QLoRA paper: "QLoRA: Efficient Finetuning of Quantized LLMs" https://arxiv.org/abs/2305.14314

#### Containerization with Podman (Red Hat tooling)
- Podman: Docker-compatible, daemonless, rootless container engine — the Red Hat standard
- `podman build`, `podman run`, `podman push` — same CLI as Docker; most Dockerfiles work unchanged
- Podman Compose: `podman-compose` for multi-container dev environments (replaces docker-compose)
- Building ML container images: choosing base images (`pytorch/pytorch`, `nvidia/cuda`, `ubi9/python-311`)
- Multi-stage builds for smaller production images
- Image registry: Quay.io (Red Hat's registry) vs Docker Hub

#### OpenShift AI / Open Data Hub (ODH) (weeks 3-4)
- Architecture: data science projects, workbenches (JupyterHub), model servers, pipelines
- ODH components: MLflow, Ray, KubeFlow Pipelines (Data Science Pipelines), ModelMesh, single-model servers, Elyra
- OpenShift CLI: `oc` (OpenShift) and `kubectl` — differences and when to use each
- Data Science Pipelines: OpenShift AI's managed KFP, integrated with the UI

#### vLLM (week 4)
- Architecture: PagedAttention + continuous batching — key to high throughput
- Deployment on OpenShift: `Deployment`, GPU resource requests (`nvidia.com/gpu: 1`), `Service`, `Route`
- OpenAI-compatible API (`/v1/chat/completions`) — any OpenAI SDK client works with zero changes
- Quantization: GPTQ and AWQ model loading
- Multi-GPU: `--tensor-parallel-size N` for models exceeding single GPU VRAM
- Performance tuning: `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`
- Serving LoRA adapters: `--lora-modules` flag to serve multiple adapters on one base model
- vLLM + LlamaStack: configuring a LlamaStack distribution to use a vLLM provider

#### KubeFlow Pipelines v2 / Data Science Pipelines (weeks 5-6)
- KFP v2 SDK: `@dsl.component`, `@dsl.pipeline` decorators
- Component I/O: `Input[Dataset]`, `Output[Model]`, `Output[Artifact]` typed parameters
- Lightweight Python components vs. container components — when to use each
- Pipeline compilation: `compiler.Compiler().compile(pipeline_func, 'pipeline.yaml')`
- Pipeline runs, experiments, and recurring schedules via UI or Python client
- S3/Object Storage integration: MinIO (OpenShift AI default), AWS S3 — for artifacts between stages
- Pipeline caching: skip re-running unchanged upstream steps

#### Ray (weeks 6-7)
- Ray Core: `@ray.remote` functions and classes (actors); `ray.get()`, `ray.put()`
- Ray Data: distributed data preprocessing — `ray.data.read_parquet()`, `.map_batches()`, `.filter()` — vectorized operations at scale
- Ray Train: `TorchTrainer` for distributed PyTorch; `ScalingConfig(num_workers=4, use_gpu=True)`; checkpoint integration
- Ray Serve: `@serve.deployment` for scalable model serving; autoscaling policy; multiple replicas
- Ray on OpenShift: KubeRay operator installation; `RayCluster` custom resource; `RayJob` for batch workloads
- `RayCluster` spec: head node, worker nodes, resource requests, auto-scaler config

#### MLOps (week 8)
- MLflow: `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.log_artifact()`; Model Registry with stage transitions (Staging → Production)
- Prometheus + Grafana on OpenShift: `ServiceMonitor` CRD scrapes vLLM `/metrics` endpoint; key metrics: `vllm:request_throughput`, `vllm:token_throughput`, GPU utilization
- A/B testing: OpenShift Service Mesh (Istio) traffic splitting for model variants; compare RAGAS scores across variants
- CI/CD for ML: Tekton pipelines (OpenShift native) to trigger KFP runs on data changes; promote model versions on quality gate pass
- Model drift detection concepts: distribution shift, embedding drift, output quality degradation signals

### Resources

**HuggingFace**
- HuggingFace `transformers` docs: https://huggingface.co/docs/transformers
- HuggingFace `datasets` docs: https://huggingface.co/docs/datasets
- HuggingFace `PEFT` docs: https://huggingface.co/docs/peft
- HuggingFace `trl` docs: https://huggingface.co/docs/trl
- HuggingFace `accelerate` docs: https://huggingface.co/docs/accelerate
- DeepLearning.AI "Finetuning Large Language Models" (free): https://learn.deeplearning.ai/courses/finetuning-large-language-models
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314

**Podman and Containers**
- Podman docs: https://docs.podman.io/en/latest/
- Podman Compose: https://github.com/containers/podman-compose
- Red Hat UBI (Universal Base Images): https://catalog.redhat.com/software/containers/explore

**OpenShift AI / ODH**
- OpenShift AI docs: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed
- Open Data Hub docs: https://opendatahub.io/docs/ | GitHub: https://github.com/opendatahub-io
- OpenShift CLI (`oc`) reference: https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html

**vLLM**
- vLLM docs: https://docs.vllm.ai/en/latest/ | GitHub: https://github.com/vllm-project/vllm
- PagedAttention paper: https://arxiv.org/abs/2309.06180
- vLLM LoRA serving docs: https://docs.vllm.ai/en/latest/models/lora.html

**KubeFlow Pipelines**
- KFP v2 SDK docs: https://www.kubeflow.org/docs/components/pipelines/v2/
- KFP v2 GitHub: https://github.com/kubeflow/pipelines

**Ray**
- Ray docs: https://docs.ray.io/en/latest/
- KubeRay operator: https://github.com/ray-project/kuberay
- Ray Data docs: https://docs.ray.io/en/latest/data/data.html
- Ray Serve docs: https://docs.ray.io/en/latest/serve/index.html

**MLOps**
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Red Hat Developer YouTube: OpenShift AI demos

### Hands-on Projects
1. **HuggingFace Fine-tuning (LoRA)**: Fine-tune Llama 3.2 3B on a small instruction dataset using QLoRA + TRL `SFTTrainer`; compare base vs. fine-tuned on 10 held-out prompts; push adapter to HuggingFace Hub
2. **Podman AI Service**: Containerize the Phase 3 RAG FastAPI service with Podman; use multi-stage build; push to Quay.io; run with Podman Compose (app + PostgreSQL + pgvector)
3. **vLLM Deployment**: Deploy Llama 3.1 8B with vLLM on Kubernetes (Kind locally); test OpenAI-compatible API; benchmark throughput with 10/50/100 concurrent requests; serve a LoRA adapter with `--lora-modules`
4. **KFP Pipeline**: 4-stage pipeline — data ingest → Docling parsing → embedding generation (with Ray) → pgvector population; S3 artifacts between stages; run in OpenShift AI Data Science Pipelines
5. **Ray Distributed Processing**: Ray Data for 10K-document corpus preprocessing and embedding; Ray Serve autoscaling embedding model with multiple replicas; test autoscaling behavior
6. **MLflow + Prometheus**: Instrument the full RAG pipeline with MLflow tracking; add Prometheus metrics to FastAPI; set up Grafana dashboard showing request rate, p95 latency, and RAG quality score

### Capstone
**"Production RAG Platform on OpenShift AI"** — End-to-end:
- Docling-based KFP pipeline (ingest → parse → embed → pgvector + Qdrant)
- vLLM serving a QLoRA fine-tuned Llama model (from Project 1)
- Ray for distributed embedding generation at scale
- LlamaStack distribution backed by the vLLM endpoint
- MLflow tracking all pipeline runs with a quality gate
- Prometheus + Grafana metrics dashboard
- A/B test two retrieval strategies (dense vs. hybrid) on a live endpoint
- Fully containerized with Podman; deployment manifests for OpenShift AI

---

## Phase 9: GraphRAG, RAFT, InstructLab, and Multi-Modal Awareness
**Duration: 6 weeks**

### Objectives
- Implement GraphRAG for knowledge-graph-enriched retrieval
- Use RAFT to create domain-adapted fine-tuned models (builds on Phase 8 LoRA/QLoRA skills)
- Use InstructLab to contribute skills and knowledge to LLMs via synthetic data generation
- Gain practical awareness of multi-modal (vision-language) models and when they apply
- Know when each technique is worth the cost vs. vanilla RAG

### Key Concepts

#### Neo4j and Knowledge Graph Basics (prerequisite for GraphRAG)
- Graph data model: nodes, relationships, properties — vs. tables and documents
- Cypher query language basics: `MATCH`, `CREATE`, `MERGE`, `WHERE`, `RETURN`
- Key Cypher patterns: find nodes by label, traverse relationships, aggregate
- Neo4j Desktop for local development; AuraDB for cloud; Neo4j in Docker/Podman
- LangChain `Neo4jGraph` and `GraphCypherQAChain` integration
- When knowledge graphs add value: entity-rich domains (biomedical, legal, product catalogs), multi-hop reasoning needs, relationship-heavy queries

#### GraphRAG (Microsoft)
- Why graph-based RAG improves on vector-only RAG: captures entity relationships, enables multi-hop reasoning, provides global thematic summaries
- Microsoft GraphRAG indexing pipeline:
  1. Entity extraction: LLM reads every chunk, extracts named entities and relationships
  2. Graph construction: entities as nodes, relationships as edges, properties from text
  3. Community detection: Leiden algorithm groups densely connected entities
  4. Community report generation: LLM summarizes each community — this enables "global" search
- Search modes:
  - **Local search**: start from a specific entity, traverse neighbors, answer entity-anchored questions
  - **Global search**: query over community reports, answer broad thematic questions ("What are the main themes across all documents?")
- Cost reality check: indexing 1000 documents may cost $5-20 in LLM API calls; factor this into architecture decisions
- GraphRAG vs. naive RAG decision framework: use GraphRAG when queries require multi-hop reasoning or global synthesis; use naive RAG for direct fact lookup
- Alternative: **LightRAG** (simpler, faster graph construction): https://github.com/HKUDS/LightRAG

#### RAFT (Retrieval-Augmented Fine-Tuning)
Requires Phase 8 LoRA/QLoRA knowledge. RAFT is a specific fine-tuning recipe, not a new technique.

- RAFT motivation: even with good retrieval, a general-purpose model may not know how to reason over your domain's documents; RAFT teaches it
- Training data construction:
  - For each training question: include 1 oracle document (contains the answer) + K distractor documents (don't contain the answer)
  - 80% of training examples include the oracle doc; 20% omit it (teaches model to answer from memory too)
  - Target output: chain-of-thought reasoning trace that cites the oracle doc, then the final answer
- Fine-tuning: use Phase 8's QLoRA + TRL `SFTTrainer` pipeline; chat-formatted dataset
- Evaluation: RAGAS scores of RAFT-tuned model + RAG vs. base model + same RAG
- RAFT GitHub script for dataset generation: https://github.com/ShishirPatil/gorilla/tree/main/raft

#### InstructLab
- InstructLab motivation: democratize contribution to open-source LLMs — anyone can contribute domain knowledge or skills without needing ML expertise
- **Taxonomy structure**: a Git repository of YAML files
  - `knowledge/`: factual domain knowledge (Q&A pairs grounded in a document)
  - `skills/`: compositional skills (formatting, reasoning, tasks that generalize)
- **Synthetic Data Generation (SDG)**: `ilab data generate` — teacher model (Mixtral/Merlinite) reads your YAML seed examples and generates hundreds of variations
- **LAB training** (`ilab model train`):
  - Phase 1: knowledge training on SDG-generated data
  - Phase 2: skills training
  - Knowledge replay: mix in general data to prevent catastrophic forgetting
- **`ilab` CLI workflow**: `ilab config init` → `ilab data generate` → `ilab model train` → `ilab model evaluate` → `ilab model serve`
- Running full training: requires significant GPU; for learning, use `--num-epochs 1` and a small model
- RHEL AI: Red Hat's enterprise packaging of InstructLab with optimized hardware support
- OpenShift AI integration: `InstructLab` operator for training at scale on OpenShift GPU nodes
- Evaluation: `ilab model evaluate` — runs MT-Bench, MMLU subset; custom eval with `lm-evaluation-harness`

#### Multi-Modal Awareness (practical overview, not deep dive)
Multi-modal models process both text and images — increasingly relevant for document AI and enterprise use cases.

- **Vision-Language Models (VLMs)**: LLaVA, Llama 3.2 Vision, Qwen-VL, PaliGemma — accept image + text, output text
- **Use cases**: OCR post-processing, chart/diagram understanding, product image Q&A, visual document analysis
- **Multi-modal RAG**: extract images from PDFs (Docling handles this), embed image descriptions, retrieve by visual content
- Running VLMs locally: `ollama pull llama3.2-vision` — then pass base64-encoded images in messages
- LangChain multi-modal message format: `HumanMessage(content=[{"type": "image_url", ...}, {"type": "text", ...}])`
- Docling's image extraction: automatically extracts figures and generates captions using a VLM
- When to invest further: if your documents are image-heavy (engineering drawings, financial charts, medical imaging), invest a dedicated sprint; if mostly text, this awareness is sufficient

### Resources

**Neo4j and Knowledge Graphs**
- Neo4j docs: https://neo4j.com/docs/
- Cypher manual: https://neo4j.com/docs/cypher-manual/current/
- Neo4j + LangChain integration: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
- Neo4j in Docker: https://hub.docker.com/_/neo4j

**GraphRAG**
- GraphRAG paper: https://arxiv.org/abs/2404.16130
- GraphRAG GitHub: https://github.com/microsoft/graphrag
- GraphRAG docs: https://microsoft.github.io/graphrag/
- LightRAG GitHub: https://github.com/HKUDS/LightRAG

**RAFT**
- RAFT paper: https://arxiv.org/abs/2403.10131
- RAFT GitHub: https://github.com/ShishirPatil/gorilla/tree/main/raft
- HuggingFace TRL docs (SFTTrainer): https://huggingface.co/docs/trl/sft_trainer

**InstructLab**
- InstructLab docs: https://docs.instructlab.ai/
- InstructLab GitHub: https://github.com/instructlab/instructlab
- InstructLab taxonomy: https://github.com/instructlab/taxonomy
- LAB paper: https://arxiv.org/abs/2403.01081
- RHEL AI docs: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai
- `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness

**Multi-Modal**
- Llama 3.2 Vision on Ollama: `ollama pull llama3.2-vision`
- Docling image extraction: https://ds4sd.github.io/docling/usage/
- LangChain multi-modal messages: https://python.langchain.com/docs/how_to/multimodal_inputs/
- DeepLearning.AI "Multimodal RAG: Chat with Videos" (check availability): https://learn.deeplearning.ai/

### Hands-on Projects
1. **Neo4j + Cypher Basics**: Model a small domain (e.g., software project: services, APIs, teams, dependencies) in Neo4j; write 10 Cypher queries covering CREATE, MATCH, multi-hop traversal, aggregation; query via LangChain `GraphCypherQAChain`
2. **GraphRAG vs. Naive RAG**: Run GraphRAG indexing pipeline on 500 domain articles (Kubernetes docs or similar); compare local search and global search quality to Phase 3 naive RAG using the RAGAS harness; measure cost
3. **RAFT Dataset + Fine-tuning**: Generate a RAFT training dataset for a domain using the RAFT script; fine-tune Llama 3.2 3B with QLoRA + TRL (Phase 8 pipeline); evaluate fine-tuned model + RAG vs. base model + RAG with RAGAS
4. **InstructLab Contribution**: Install `ilab`; write a knowledge contribution (10+ Q&A pairs with a source document) for a domain you own; `ilab data generate`; inspect synthetic data quality; run abbreviated training; evaluate with `ilab model evaluate`
5. **Multi-Modal Document Q&A**: Use Docling to extract images and tables from a PDF; pass extracted images to `llama3.2-vision` via Ollama for description; build a RAG pipeline that includes both text chunks and image descriptions as retrievable content

### Capstone
**"Domain-Adaptive Knowledge System"** — Choose one specific domain (e.g., OpenShift operations, automotive regulations, or mobility telematics). Build and compare all four approaches:
1. Naive RAG (Phase 3 pipeline, used as baseline)
2. GraphRAG (Neo4j + Microsoft GraphRAG indexing)
3. RAFT-tuned model (QLoRA fine-tuned on RAFT dataset, served via vLLM from Phase 8)
4. InstructLab-contributed knowledge (ilab workflow)

Evaluate all four on a 50-question held-out eval set using RAGAS. Package findings as an **Architectural Decision Record (ADR)** in `projects/phase9-domain-adaptive/` recommending when to use each approach, with supporting evaluation data.

---

## Final Capstone: Enterprise Knowledge Assistant Platform
**Duration: 2 weeks**

Combines every phase into a production-architected system:

| Component | Technology |
|-----------|-----------|
| Document Ingestion | Docling (parsing) → KubeFlow Pipelines on OpenShift AI |
| Embedding + Indexing | Ray Data (distributed) → pgvector + Qdrant |
| Model Serving | vLLM on OpenShift AI (serving RAFT or InstructLab fine-tuned model) |
| Retrieval | pgvector dense + BM25 sparse + GraphRAG global + cross-encoder reranker |
| Caching | Redis Semantic Cache |
| Orchestration | LangGraph multi-agent: supervisor → research → analysis → writing |
| Tool Layer | Custom MCP server; agents consume via MCP protocol |
| Safety | LlamaGuard + NeMo Guardrails or Guardrails AI |
| Provider Abstraction | LlamaStack distribution → vLLM backend |
| Observability | LangSmith tracing + MLflow model registry + Prometheus/Grafana |
| Evaluation | RAGAS scorecard + LangSmith eval datasets + human-in-the-loop feedback |
| Containerization | Podman; deployment manifests for OpenShift AI |

**Deliverables:**
- Architecture diagram (component view + data flow view)
- Architectural Decision Records (ADRs) for key choices (vector DB, fine-tuning approach, guardrails strategy)
- Fully containerized deployment manifests for OpenShift AI
- RAGAS evaluation report comparing all retrieval strategies
- Runbook for model updates: InstructLab → vLLM hot-swap procedure

---

## Workspace Structure

```
.
├── LEARNING_PATH.md                  # This document (living reference)
├── projects/
│   ├── phase2-langchain/             # Research assistant agent + streaming API
│   ├── phase3-rag/                   # Technical documentation search (Docling + pgvector)
│   ├── phase4-advanced-rag/          # Advanced retrieval + semantic cache
│   ├── phase5-langgraph/             # Autonomous research pipeline
│   ├── phase6-mcp-guardrails/        # MCP server + guardrailed agent
│   ├── phase7-llamastack/            # Portable AI application
│   ├── phase8-openshift/             # Production RAG platform on OpenShift AI
│   └── phase9-domain-adaptive/       # Domain-adaptive knowledge system + ADR
├── notebooks/                        # Jupyter notebooks for experimentation
└── evals/                            # Reusable RAGAS evaluation harnesses
```

---

## Ongoing Reference Resources

### Key YouTube Channels
- **Andrej Karpathy** — LLMs and neural networks from first principles
- **Sam Witteveen** (@samwitteveenai) — LangChain, agents, practical tutorials
- **AI Explained** — Research paper breakdowns for practitioners
- **Yannic Kilcher** — Deep ML paper reviews
- **Red Hat Developer** — OpenShift AI, InstructLab, and RHEL AI demos

### DeepLearning.AI Short Courses (free, 1-3 hours each)
All at https://learn.deeplearning.ai/
- LangChain for LLM Application Development
- LangChain: Chat with Your Data
- Building and Evaluating Advanced RAG
- AI Agents in LangGraph
- Finetuning Large Language Models
- Building Agentic RAG with LlamaIndex
- Multimodal RAG: Chat with Videos (check availability)

### Recommended Books
- *Hands-On Large Language Models* — Jay Alammar & Maarten Grootendorst (O'Reilly 2024)
- *AI Engineering* — Chip Huyen (O'Reilly 2025) — production ML systems; strongly recommended
- *Building LLM Powered Applications* — Valentina Alto (O'Reilly 2024)
- *Designing Machine Learning Systems* — Chip Huyen — MLOps foundation

### Key Papers (Reading List by Phase)
- Phase 1: "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
- Phase 2: ReAct (2022) — https://arxiv.org/abs/2210.03629
- Phase 3: RAGAS (2023) — https://arxiv.org/abs/2309.15217
- Phase 4: HyDE (2022) — https://arxiv.org/abs/2212.10496 | CRAG (2024) — https://arxiv.org/abs/2401.15884
- Phase 6: "Lost in the Middle" (2023) — https://arxiv.org/abs/2307.03172 | LlamaGuard — https://arxiv.org/abs/2312.06674
- Phase 8: LoRA (2021) — https://arxiv.org/abs/2106.09685 | QLoRA (2023) — https://arxiv.org/abs/2305.14314 | PagedAttention — https://arxiv.org/abs/2309.06180
- Phase 9: GraphRAG (2024) — https://arxiv.org/abs/2404.16130 | RAFT (2024) — https://arxiv.org/abs/2403.10131 | LAB (2024) — https://arxiv.org/abs/2403.01081
