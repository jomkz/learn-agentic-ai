# ADR-003: Agent Orchestration Framework

**Status**: Accepted
**Date**: 2026-01-15
**Deciders**: Platform team

---

## Context

The platform requires an agent orchestration framework to coordinate a multi-agent pipeline:
supervisor routing, specialist agents (research, writing, calculation, verification), tool
invocation via MCP, multi-turn conversational state, and human-in-the-loop checkpoints.

Requirements:
- Stateful multi-turn conversations with persistent graph state
- Conditional routing between agents based on query intent and intermediate results
- Human-in-the-loop (HIL) support: pause graph execution and await human approval at
  designated edges (used for high-stakes synthesis before delivery to regulated users)
- Retry logic and fallback routing when an agent returns a low-confidence result
- Production durability: graph state must survive application pod restarts
- Observable: all node inputs/outputs, tool calls, and latencies must be traceable
- Python-native: must integrate with the existing LangChain/LangSmith toolchain

Frameworks evaluated:

| Framework | Checkpointing | HIL | Conditional routing | Maturity |
|-----------|--------------|-----|---------------------|----------|
| LangGraph | Yes (Postgres) | Yes (interrupt_before/after) | Yes (conditional_edge) | Production |
| LlamaStack native agents | Limited | No | Limited | Beta |
| CrewAI | Limited (in-memory) | No | Role-based only | Growing |
| AutoGen | Yes (planned) | Partial | Yes | Research-origin |

---

## Decision

**LangGraph for agent orchestration. LlamaStack as the inference and safety substrate.**

These are complementary, not competing: LangGraph defines the graph topology and manages state;
LlamaStack provides the inference API, safety shields, and tool-call protocol that LangGraph
nodes call into.

### LangGraph

LangGraph is selected as the orchestration layer for the following reasons:

**Checkpointing and durability**: LangGraph's checkpointer interface, backed by PostgreSQL (the
same instance used for application metadata), persists the full `AgentState` TypedDict at every
node boundary. A pod restart resumes from the last checkpoint transparently. This is the single
most important production durability property for a stateful multi-agent system.

**Human-in-the-loop**: `interrupt_before` and `interrupt_after` decorators on any node edge
pause graph execution and surface the current state to an external review interface. The platform
uses this for high-stakes document synthesis in regulated contexts. No other evaluated framework
provides first-class HIL support.

**Conditional routing**: `add_conditional_edges` with a Python callable gives full programmatic
control over routing decisions. The supervisor node classifies query intent and routes to the
appropriate specialist agent(s). If the research agent returns `confidence < 0.7`, a conditional
edge routes to the verification agent before the writing agent is invoked. This routing logic is
plain Python and is fully testable in isolation.

**LangSmith integration**: LangGraph traces are forwarded to LangSmith automatically via the
`LANGCHAIN_TRACING_V2=true` environment variable. Node-level latencies, token counts, and tool
call inputs/outputs are visible in the LangSmith UI without additional instrumentation.

**Production maturity**: LangGraph is used in production by multiple large enterprises. The API
is stable, the documentation is thorough, and the LangChain team provides long-term support.

### LlamaStack as inference/safety substrate

LlamaStack provides:
- A unified inference API over vLLM (and other backends) with a consistent interface
- Safety shields (backed by LlamaGuard) invoked as part of the inference call
- Tool-call protocol compatible with MCP

LangGraph nodes call the LlamaStack inference API via its Python client. This means that
switching the inference backend (e.g., from vLLM to a hosted API) requires only a LlamaStack
configuration change, not a LangGraph graph change.

### Options not selected

**LlamaStack native agents (standalone)**: LlamaStack's built-in agent loop supports tool calls
and basic ReAct-style reasoning, but does not provide a graph abstraction, persistent
checkpointing, or conditional routing between multiple specialist agents. It is the right choice
for single-agent tool-use scenarios but cannot satisfy the multi-agent supervisor requirements
of this platform. Using LlamaStack as the inference substrate while orchestrating with LangGraph
combines the strengths of both.

**CrewAI**: CrewAI's role-based crew abstraction is intuitive for straightforward pipelines but
lacks production checkpointing (state is in-memory and lost on pod restart) and does not support
HIL interrupts. The framework is also opinionated about how agents communicate, which conflicts
with the platform's need for custom routing logic in the supervisor.

**AutoGen**: AutoGen originated as a research framework and its production-readiness has improved,
but the API surface has changed significantly across versions. The conversation-based agent
model makes conditional routing based on intermediate results more complex than in LangGraph's
explicit edge model. AutoGen remains a strong candidate to re-evaluate if LangGraph's performance
under high concurrency proves to be a bottleneck.

---

## Consequences

### What becomes easier

- Full graph state (all agent inputs/outputs, tool calls, conversation history) is persisted to
  PostgreSQL. Debugging a failed or unexpected run is a database query away.
- HIL checkpoints are first-class: product, compliance, and legal reviewers can be integrated
  into the agent workflow without hacking around the framework.
- Swapping the inference backend (vLLM version update, model change) is a LlamaStack config
  change only; the LangGraph graph topology is unchanged.
- Unit testing individual nodes and routing logic is straightforward: nodes are Python callables
  that accept and return `AgentState` TypedDicts.

### What becomes harder

- LangGraph's StateGraph abstraction has a learning curve steeper than simple chain-based
  frameworks. New team members need to understand the node/edge/state mental model before
  contributing to the graph topology.
- Debugging complex graph topologies requires LangSmith; local debugging without tracing is
  difficult for graphs with more than three nodes. Teams must treat LangSmith as a required
  dependency, not an optional observability add-on.
- The two-layer architecture (LangGraph + LlamaStack) means that stack traces for inference
  errors cross two framework boundaries, which can make root-cause analysis slower.
- Long-running graph executions (verification loops, HIL pauses) accumulate checkpoint rows in
  PostgreSQL. A checkpoint retention policy and periodic cleanup job are required to prevent
  unbounded table growth.
