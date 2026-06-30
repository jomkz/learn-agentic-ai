# Phase 5: LangGraph — Stateful Agents and Multi-Agent Workflows

**Duration: 4 weeks** | [← Phase 4](04-advanced-rag.md) | [Phase 6 →](06-agentic-patterns-mcp.md)

**Project directory:** [`projects/phase5-langgraph/`](../../projects/phase5-langgraph/)

---

## Objectives

- Build complex, stateful agent workflows with checkpointing and fault tolerance
- Implement human-in-the-loop approval and review flows with streaming
- Design production multi-agent architectures for real business problems

---

## Key Concepts

### StateGraph Fundamentals
- `StateGraph`: nodes (Python functions or runnables), edges (always-on transitions), conditional edges (router function decides next node)
- `END`: terminal node — graph stops here
- State management:
  - `TypedDict`: simple, fast, no validation overhead
  - Pydantic `BaseModel`: validated state with type checking — use for production
- Compiled graphs: `graph.compile()` → `CompiledGraph`
- Invocation modes: `graph.invoke()` (blocking), `graph.stream()` (step-by-step), `graph.astream_events()` (fine-grained events)

### Checkpointing and Persistence
- Checkpointers write full graph state after every node execution
  - `MemorySaver`: in-memory, lost on restart — dev only
  - `SqliteSaver`: SQLite file — persistent dev/testing
  - `PostgresSaver`: production — scales, concurrent, durable
- Thread IDs: every conversation/session gets a unique thread ID; state is isolated per thread
- Time travel: replay or branch from any prior checkpoint — essential for debugging
- Resume after failure: if a node crashes, re-invoke with the same thread ID to resume from last checkpoint

### Human-in-the-Loop
- `interrupt_before=["node_name"]`: pause before executing a node; wait for human input
- `interrupt_after=["node_name"]`: pause after; inspect output before continuing
- `Command(resume=value)`: resume graph with a human-supplied value
- Use cases: approval workflows, content review, dangerous action confirmation

### Streaming Modes
- `"values"`: emit full state dict after each node — easy to display current state
- `"updates"`: emit only the state changes per node — lower bandwidth
- `"debug"`: emit everything including internal events — use for debugging only

### Multi-Agent Patterns

**Supervisor**
- A router node calls an LLM to decide which specialist agent to invoke next
- Specialists return to the supervisor after completing their task
- Best for: well-defined task taxonomy, clear specialist domains

**Hierarchical**
- Supervisors of supervisors — each supervisor manages a sub-team
- Best for: very large agent systems with multiple domains

**Swarm / Handoffs**
- Agents pass control directly: `Command(goto="other_agent", update={...})`
- No central coordinator; agents decide who handles next
- Best for: fluid, context-dependent task routing

### Long-Term Memory
- `InMemoryStore` → `PostgresStore`: stores facts that persist across threads (not just within one conversation)
- Use for: user preferences, learned facts, cross-session context

### Multi-Agent Framework Landscape

LangGraph is not the only multi-agent framework. As an architect you will encounter these alternatives and need to know when each is appropriate.

| Framework | Model | Strengths | Weaknesses | When to use |
|-----------|-------|-----------|------------|-------------|
| **LangGraph** | Graph with stateful nodes | Full control: custom topology, checkpointing, human-in-the-loop, streaming, production-grade | More code to write; steeper learning curve | Production systems needing durability, custom routing, or HIL |
| **CrewAI** | Role-based crew of agents | Fast to get started; YAML-defined agents and tasks; built-in role/backstory/goal system | Less control over graph topology; limited checkpointing | Prototypes, role-oriented workflows, demos |
| **AutoGen** (Microsoft) | Conversation-driven agents | Flexible conversational multi-agent patterns; good research tool; active community | Less production-hardened than LangGraph; Microsoft ecosystem bias | Research, flexible conversation topologies, teams already using Azure OpenAI |
| **Semantic Kernel** (Microsoft) | Plugin-based orchestration | Strong .NET/C# story; enterprise Microsoft ecosystem | Python support secondary; opinionated enterprise patterns | Microsoft/Azure shops, .NET backends |

**Key architectural takeaway:** LangGraph gives you the most control over state, routing, and failure recovery — which is why it's the right choice for production enterprise agents. CrewAI and AutoGen are faster for prototyping and exploration; they are not substitutes for production durability requirements.

**Pattern to know — CrewAI quick comparison:**
```python
# CrewAI style — declarative, role-based
from crewai import Agent, Task, Crew

researcher = Agent(role="Research Analyst", goal="Find accurate information", ...)
writer = Agent(role="Technical Writer", goal="Synthesize findings clearly", ...)
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()

# LangGraph equivalent — imperative, stateful, checkpointed
graph = StateGraph(ResearchState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_edge("researcher", "writer")
app = graph.compile(checkpointer=PostgresSaver(...))
result = app.invoke({"query": "..."}, config={"thread_id": "123"})
```

---

## Resources

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangGraph tutorials: https://langchain-ai.github.io/langgraph/tutorials/
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- DeepLearning.AI "AI Agents in LangGraph" (free): https://learn.deeplearning.ai/courses/ai-agents-in-langgraph
- Plan-and-Solve paper: https://arxiv.org/abs/2305.04091
- Autonomous Agents survey: https://arxiv.org/abs/2308.11432
- CrewAI docs: https://docs.crewai.com/
- CrewAI GitHub: https://github.com/crewAIInc/crewAI
- AutoGen docs: https://microsoft.github.io/autogen/
- AutoGen GitHub: https://github.com/microsoft/autogen

---

## Hands-on Projects

1. **Stateful Research Agent** — Rebuild the Phase 2 Research Agent in LangGraph with:
   - `PostgresSaver` checkpointing (persistent across restarts)
   - Human approval interrupt before web searches
   - `astream_events` output piped to the Phase 2 SSE endpoint
   - Time-travel debugging: replay from a prior state to inspect a bad output

2. **Multi-Agent Code Review Pipeline** — Supervisor routes code review tasks to specialized subagents (security, performance, style); each streams findings; supervisor synthesizes a final report with overall verdict

3. **Human-in-the-Loop Content Workflow** — Agent drafts content → human approves or edits in-place → agent revises and "publishes"; full interrupt/resume/state-persistence cycle

4. **Framework Comparison** — Implement the same two-agent (research + write) workflow in both CrewAI and LangGraph; document the tradeoffs: lines of code, checkpointing support, streaming support, observability; write a 1-page ADR recommending which to use in a production enterprise context and why

### Capstone: Autonomous Research Pipeline
Multi-agent system where a supervisor orchestrates:
- **Research agent**: web search + RAG from Phase 3
- **Data analysis agent**: Python code interpreter (runs code, returns results)
- **Writing agent**: synthesizes findings into a structured report

Requirements: `PostgresSaver` checkpointing, streams via SSE endpoint, LangSmith traces show full graph execution, evaluation compares output quality to Phase 2 single-agent baseline.

---

## Completion Checklist

- [ ] `StateGraph` with conditional edges routes between at least 3 nodes correctly
- [ ] `PostgresSaver` persists state — kill the process, restart, resume from same thread ID
- [ ] Human approval interrupt pauses execution; `Command(resume=...)` continues it
- [ ] Time travel: invoke graph from a prior checkpoint and get the same intermediate output
- [ ] Supervisor correctly routes to ≥3 different specialist agents based on task type
- [ ] Swarm pattern: at least one agent uses `Command(goto=...)` to hand off directly
- [ ] Capstone pipeline runs end-to-end with all three agents and produces a report
- [ ] LangSmith shows the full multi-agent trace with per-node timing
- [ ] CrewAI comparison: same workflow implemented in CrewAI and LangGraph; ADR documents the tradeoffs with specific, observable differences
