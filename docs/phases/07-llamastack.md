# Phase 7: LlamaStack

**Duration: 3 weeks** | [← Phase 6](06-agentic-patterns-mcp.md) | [Phase 8 →](08-huggingface-openshift.md)

**Project directory:** [`projects/phase7-llamastack/`](../../projects/phase7-llamastack/)

---

## Objectives

- Understand LlamaStack's provider abstraction and distribution model
- Build applications and agentic loops using LlamaStack's native APIs
- Understand how LlamaStack complements LangChain/LangGraph in a production architecture

---

## Key Concepts

### What LlamaStack Is (and Isn't)

LlamaStack is an **infrastructure layer and API standard** — it defines *how* you serve models, attach tools, and enforce safety. LangChain/LangGraph is an **orchestration layer** — it defines *how* you chain calls and build graphs.

They compose cleanly: **LangGraph as the brain, LlamaStack as the model-serving and safety substrate.**

Think of LlamaStack like the Linux kernel: it standardizes the AI application stack so that code written against the LlamaStack API works with any provider — local Ollama, Together AI, vLLM on OpenShift — by swapping the distribution config, not the application code.

### Architecture

```
Your Application Code
        ↓
LlamaStack Python Client  (same API regardless of provider)
        ↓
LlamaStack Server  (a "distribution" — a configured bundle of providers)
        ↓
Providers: Ollama | Together | Fireworks | vLLM | custom
```

**Distributions** are pre-configured bundles of providers. Key distributions:
- `ollama`: local dev — no cloud dependency
- `together`: Together AI hosted inference
- `fireworks`: Fireworks AI hosted inference
- Custom: point to any OpenAI-compatible server (e.g., vLLM on OpenShift from Phase 8)

### Core APIs

| API | Purpose |
|-----|---------|
| **Inference** | Chat completions, streaming, tool calling |
| **Safety** | Shields, LlamaGuard content filtering |
| **Memory** | Memory banks: in-memory, vector, key-value |
| **Agents** | Built-in agentic loop with tool groups and sessions |
| **DatasetIO** | Dataset management for evals and fine-tuning |
| **Scoring** | Evaluation and benchmarking |

### LlamaStack Agents
The native agentic loop handles multi-turn sessions, tool calling, and memory retrieval without requiring LangGraph. Use it when you want a simple agentic flow without custom graph logic.

- **Tool groups**: built-in (`web_search`, `code_interpreter`, `memory`) and custom
- **Sessions**: isolated conversation contexts with their own tool history
- **Memory integration**: agents can store and retrieve from memory banks mid-conversation
- Limitation vs. LangGraph: no conditional routing, no custom graph topology, no human-in-the-loop interrupts — use LangGraph for those

### Safety with LlamaGuard
- Configure a `llama_guard` shield in the distribution YAML
- Shield runs on every inference call — input and/or output
- Returns a safety score; blocked responses are returned as an error
- Categories: violence, hate speech, sexual content, criminal planning, etc.

### Running Locally
```bash
# Install and start with Ollama provider
pip install llama-stack
llama stack build --template ollama --image-type conda
llama stack run ollama

# Then in Python:
from llama_stack_client import LlamaStackClient
client = LlamaStackClient(base_url="http://localhost:5001")
```

---

## Resources

- LlamaStack docs: https://llama-stack.readthedocs.io/en/latest/
- LlamaStack GitHub: https://github.com/meta-llama/llama-stack
- LlamaStack Python client: https://github.com/meta-llama/llama-stack-client-python
- LlamaStack Apps examples: https://github.com/meta-llama/llama-stack-apps
- LlamaStack distributions reference: https://llama-stack.readthedocs.io/en/latest/distributions/

---

## Hands-on Projects

1. **LlamaStack + Ollama Distribution** — Configure and run a LlamaStack server with Ollama provider; write a Q&A application using the Python client; compare the development experience to the same app written with LangChain
2. **LlamaStack Agent with Safety** — Build a native LlamaStack agent with LlamaGuard shield enabled; test it blocks unsafe content; implement tool calling using the web search tool group
3. **LlamaStack RAG** — Store 50 documents in a LlamaStack vector memory bank; query the memory bank; integrate retrieval results into an agent's response
4. **LangGraph + LlamaStack Integration** — Build a LangGraph graph where each LLM node calls the LlamaStack Inference API; demonstrates LangGraph orchestration with LlamaStack serving

### Capstone: Portable AI Application
Application that runs against both:
- Local: LlamaStack Ollama distribution (Llama 3.1 8B)
- Cloud: Anthropic API via LangChain

Switching providers is controlled by an environment variable — application code is unchanged. LlamaStack handles inference + safety shields. LangGraph handles orchestration. LangSmith traces the full execution.

---

## Completion Checklist

- [ ] LlamaStack server starts with Ollama distribution and responds to `client.inference.chat_completion()`
- [ ] LlamaStack agent handles a multi-turn conversation with at least 2 tool calls
- [ ] LlamaGuard shield blocks at least one unsafe prompt (test with an explicit harmful request)
- [ ] Vector memory bank stores 50 docs and returns relevant results to a query
- [ ] LangGraph graph successfully calls LlamaStack inference API from a graph node
- [ ] Capstone: switching `LLAMA_STACK_BASE_URL` vs. `ANTHROPIC_API_KEY` env var changes the provider with no code change
