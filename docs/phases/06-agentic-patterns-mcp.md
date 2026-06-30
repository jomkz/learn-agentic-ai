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

---

## Resources

- MCP official docs: https://modelcontextprotocol.io/docs
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP GitHub (spec + community servers): https://github.com/modelcontextprotocol
- NeMo Guardrails GitHub: https://github.com/NVIDIA/NeMo-Guardrails
- NeMo Guardrails docs: https://docs.nvidia.com/nemo/guardrails
- Guardrails AI: https://www.guardrailsai.com/docs
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

---

## Completion Checklist

- [ ] MCP server starts via `python server.py` and exposes 3 tools over stdio
- [ ] Claude Desktop can use the MCP server tools (verify in Claude Desktop UI)
- [ ] LangGraph agent uses MCP tools via `MultiServerMCPClient` — no custom integration code
- [ ] NeMo Guardrails blocks an off-topic request (e.g., "write me a poem") from the research agent
- [ ] Guardrails AI redacts a phone number or email from agent output before it reaches the log
- [ ] Prompt injection test: attempting to override system instructions returns a refusal, not the injected instruction
- [ ] Context budget manager trims message history without dropping the most recent turn
