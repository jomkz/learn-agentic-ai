# Phase 6: Agentic Patterns, MCP, Guardrails, and Context Management

**Duration: 3 weeks** | [← Phase 5](05-langgraph.md) | [Phase 7 →](07-llamastack.md)

**Project directory:** [`projects/phase6-mcp-guardrails/`](../../projects/phase6-mcp-guardrails/)

---

## Objectives

- Master production agentic AI design patterns
- Build and consume MCP (Model Context Protocol) servers
- Add output guardrails and safety rails to production agents
- Implement reliable context window management strategies

---

## Key Concepts

### Agentic Design Patterns

| Pattern | Description | Best for |
|---------|-------------|---------|
| **ReAct** | Interleave reasoning traces with action calls | Default; most LangGraph agents use this |
| **Plan-and-Execute** | Explicitly plan steps, then execute each | Complex multi-step tasks with known structure |
| **Reflexion** | Self-critique loop — agent evaluates and retries | Quality-sensitive tasks; costs more tokens |
| **Tree of Thoughts** | Explore multiple reasoning paths, pick best | Hard reasoning problems; expensive |
| **LATS** | MCTS-style tree search over action sequences | Maximum quality; very expensive |

In practice: ReAct handles 90% of cases. Add Reflexion when output quality is critical. Reserve ToT/LATS for research or high-stakes decisions.

### Model Context Protocol (MCP)

MCP standardizes tool exposure so any LLM client can use any tool server without custom integration code.

**Architecture:**
```
MCP Host (LangChain, LlamaStack, Claude Desktop)
    └── MCP Client
            └── MCP Server (your tools, via stdio or SSE)
```

**Transports:**
- `stdio`: local subprocess — most common for local tools
- `SSE`: HTTP — for remote servers, shared infrastructure

**MCP Primitives:**
- **Tools**: callable functions with JSON Schema input definitions → `@server.tool()`
- **Resources**: readable data (files, DB rows, API responses) → `@server.resource()`
- **Prompts**: reusable prompt templates → `@server.prompt()`

**Building MCP servers in Python:**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("my-tools")

@server.tool()
async def query_database(sql: str) -> str:
    """Run a read-only SQL query and return results as CSV."""
    ...
```

**Consuming MCP servers:**
- LangChain: `MultiServerMCPClient`
- LlamaStack: tool groups pointing to MCP server URLs
- Claude Desktop: `~/.claude/claude_desktop_config.json` with `mcpServers` entries

**Community MCP servers worth knowing:** filesystem, PostgreSQL, GitHub, web fetch, Docker, Brave Search

### Agent-to-Agent (A2A) Protocol

A2A is Google's open protocol for inter-agent communication — the counterpart to MCP. Where MCP standardizes how an agent exposes **tools to a host**, A2A standardizes how **agents communicate with each other** across organizational and infrastructure boundaries.

**MCP vs. A2A — the mental model:**
- **MCP**: agent ↔ tool server (a single agent consuming capabilities)
- **A2A**: agent ↔ agent (peer communication between autonomous systems, potentially from different vendors or orgs)

**Architecture:**
```
Client Agent                         Remote Agent
    │                                     │
    │── POST /tasks/send ──────────────>  │  (JSON-RPC over HTTP)
    │<─ task response / streaming ───────  │
    │── GET /tasks/{id} ──────────────>   │  (poll or SSE)
    │<─ final artifact ───────────────    │
```

**Agent Card** — each A2A agent advertises its capabilities at `/.well-known/agent.json`:
```json
{
  "name": "Research Agent",
  "description": "Searches and summarizes technical documents",
  "capabilities": { "streaming": true, "pushNotifications": false },
  "skills": [{ "id": "search", "name": "Document Search" }]
}
```

**Key concepts:**
- **Tasks**: the unit of work — a client sends a task, a remote agent completes it asynchronously
- **Artifacts**: the outputs of a task (text, files, structured data) returned when complete
- **Streaming**: Server-Sent Events for real-time task progress updates
- **Push notifications**: webhook callbacks for long-running tasks

**In practice:** A2A is emerging — not yet as widely adopted as MCP. Learn it now because it will define how enterprise multi-agent systems communicate across service boundaries.

### Guardrails and Safety

**Defense layers (apply all three in production):**

1. **Input guardrails** — sanitize and validate incoming user messages before the LLM sees them
   - Prompt injection defense: system prompt hardening, instruction hierarchy, `[INST]` token separation
   - PII detection: redact sensitive data before logging or passing to third-party APIs

2. **Output guardrails** — validate and filter LLM outputs before acting on them or returning to users
   - Always parse agent output against a schema; never pass raw strings to downstream systems
   - **NeMo Guardrails** (NVIDIA): declarative `.co` Colang files for topic rails, fact-checking, jailbreak defense; integrates with LangChain: https://github.com/NVIDIA/NeMo-Guardrails
   - **Guardrails AI**: Python-first validators (`PiiDetection`, `ToxicLanguage`, `ValidJson`, custom); wraps any LLM call: https://www.guardrailsai.com/docs

3. **Safety classifiers** — LLM-based content classification as a separate call
   - **LlamaGuard** (Meta): open-source safety classifier; runs as a shield in LlamaStack; can be self-hosted

4. **Red-teaming and adversarial probing** — systematic testing of model safety before production
   - **Garak** (NVIDIA): open-source LLM vulnerability scanner; runs hundreds of automated probes (prompt injection, data leakage, hallucination, toxicity, jailbreaks) against any OpenAI-compatible endpoint
     ```bash
     pip install garak
     # Probe a local vLLM or Ollama endpoint
     garak --model_type openai --model_name llama3.1:8b \
           --probes promptinject,dan,knownbadsignatures
     ```
   - Garak generates a vulnerability report; use it to identify which guardrail rules to tighten
   - Run Garak as part of CI/CD before promoting a new model version to production
   - **LLM Guard**: runtime input/output scanning library — complements Garak (Garak = offline probing, LLM Guard = online filtering): https://llm-guard.com/

### Context Window Management

- Token counting before LLM calls: `tiktoken.encoding_for_model()`, HuggingFace `AutoTokenizer`
- Dynamic context truncation: prioritize recent messages + retrieved context; drop oldest conversation turns
- Summarization for long conversations: dedicated summary node in LangGraph; replace message history with summary every N turns
- The **"Lost in the Middle" problem**: models attend poorly to content in the middle of long contexts; put critical content at the start or end of the prompt
- Long-context models (Llama 3.1 128K, Gemini 1M): longer context ≠ better attention; test empirically

**Agent reliability patterns:**
- Timeout: wrap node execution with `asyncio.wait_for(coroutine, timeout=30)`
- Retry with exponential backoff: `tenacity` library
- Fallback chains: if primary model fails, route to a cheaper/different model
- Cost/token budget per agent step: track spend in state, abort if budget exceeded

### Observability: OpenTelemetry for AI

LangSmith traces agent execution within the LangChain ecosystem. OpenTelemetry is the vendor-neutral standard that lets you send traces to any backend (Jaeger, Tempo, Honeycomb, Datadog).

**Why OpenTelemetry matters for AI architects:**
- LangSmith is great during development; production systems often require traces in the organization's existing observability stack
- OpenTelemetry spans let you correlate AI agent traces with service mesh traces, database calls, and infrastructure metrics in one view
- The `opentelemetry-instrumentation-*` ecosystem is adding AI-specific semantic conventions

**Key patterns:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OTLP exporter (Tempo, Jaeger, etc.)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-agent")

with tracer.start_as_current_span("rag-retrieval") as span:
    span.set_attribute("query.text", query)
    span.set_attribute("retrieval.k", 5)
    docs = retriever.invoke(query)
    span.set_attribute("retrieval.count", len(docs))
```

**OpenLIT** — OpenTelemetry-native observability SDK for LLMs (auto-instruments LangChain, OpenAI, vLLM): https://github.com/openlit/openlit

**Semantic conventions for AI (OTel GenAI):** emerging standard for AI span attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`): https://opentelemetry.io/docs/specs/semconv/gen-ai/

**LangSmith vs. OpenTelemetry** — not either/or:
| | LangSmith | OpenTelemetry |
|---|---|---|
| Best for | LangChain-native development tracing | Production, multi-system, vendor-neutral |
| Granularity | LangChain primitives (chain, tool, LLM call) | Custom spans at any level |
| Backend | LangSmith SaaS | Any OTLP-compatible backend |
| Correlation | Agent traces only | Correlates with infra, DB, and service traces |

---

## Resources

- MCP official docs: https://modelcontextprotocol.io/docs
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP GitHub (spec + community servers): https://github.com/modelcontextprotocol
- A2A Protocol spec: https://google.github.io/A2A/
- A2A GitHub: https://github.com/google/A2A
- NeMo Guardrails GitHub: https://github.com/NVIDIA/NeMo-Guardrails
- NeMo Guardrails docs: https://docs.nvidia.com/nemo/guardrails
- Guardrails AI: https://www.guardrailsai.com/docs
- Garak GitHub: https://github.com/leondz/garak
- Garak docs: https://docs.garak.ai/
- LLM Guard: https://llm-guard.com/
- OpenLIT GitHub: https://github.com/openlit/openlit
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- ReAct paper: https://arxiv.org/abs/2210.03629
- Reflexion paper: https://arxiv.org/abs/2303.11366
- "Building Effective Agents" — Anthropic blog: https://www.anthropic.com/research/building-effective-agents

**Key papers:** "Lost in the Middle" (2023): https://arxiv.org/abs/2307.03172 | LlamaGuard: https://arxiv.org/abs/2312.06674

---

## Hands-on Projects

1. **MCP Server** — Build a Python MCP server exposing 3 tools: filesystem access (`list_files`, `read_file`), SQL DB query (read-only), HTTP API wrapper. Test with Claude Desktop and programmatically via Python MCP client
2. **MCP + LangGraph** — Integrate the MCP server into a LangGraph agent using `MultiServerMCPClient`; demonstrate the agent using MCP tools interchangeably with native LangChain tools
3. **Guardrailed Agent** — Add to the Phase 5 Research Pipeline:
   - NeMo Guardrails `.co` file blocking off-topic requests
   - Guardrails AI PII redaction on all agent outputs before logging
   - Test that prompt injection attempts are blocked
4. **Context Budget Manager** — Utility class that: counts tokens in messages, dynamically trims oldest turns to stay under budget, emits a warning when approaching limit; integrate into the Phase 5 agent

5. **A2A Agent Server** — Expose the Phase 5 Research Agent as an A2A-compatible server with an Agent Card at `/.well-known/agent.json`; build a simple A2A client that delegates a research task to it; verify task lifecycle (submit → poll → artifact retrieval) works end-to-end

6. **Garak Security Scan** — Run Garak against the local Ollama/vLLM endpoint with at least 3 probe types (`promptinject`, `dan`, `knownbadsignatures`); review the report; add or tighten a NeMo Guardrails rule to address the highest-severity finding

7. **OpenTelemetry Instrumentation** — Instrument the Phase 5 multi-agent pipeline with OpenTelemetry spans; export traces to a local Jaeger instance (`podman run jaegertracing/all-in-one`); verify that a single user query produces a trace showing all agent node executions with token counts as span attributes

---

## Completion Checklist

- [ ] MCP server starts via `python server.py` and exposes 3 tools over stdio
- [ ] Claude Desktop can use the MCP server tools (verify in Claude Desktop UI)
- [ ] LangGraph agent uses MCP tools via `MultiServerMCPClient` — no custom integration code
- [ ] NeMo Guardrails blocks an off-topic request (e.g., "write me a poem") from the research agent
- [ ] Guardrails AI redacts a phone number or email from agent output before it reaches the log
- [ ] Prompt injection test: attempting to override system instructions returns a refusal, not the injected instruction
- [ ] Context budget manager trims message history without dropping the most recent turn
- [ ] A2A Agent Card is accessible at `/.well-known/agent.json`; A2A client completes a full task lifecycle
- [ ] Garak scan completes and produces a vulnerability report; at least one finding addressed with a guardrail rule
- [ ] OpenTelemetry traces visible in Jaeger UI; span attributes include `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`
