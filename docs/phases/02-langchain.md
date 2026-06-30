# Phase 2: LangChain + LCEL + LangSmith + Streaming + Testing

**Duration: 5 weeks** | [← Phase 1](01-foundations.md) | [Phase 3 →](03-rag-fundamentals.md)

**Project directory:** [`projects/phase2-langchain/`](../../projects/phase2-langchain/)

---

## Objectives

- Build LLM applications using LCEL pipe composition
- Implement agents with tools, memory, and multi-turn conversations
- Use LangSmith for tracing, debugging, and systematic evaluation
- Produce streaming AI responses and structure reliable tool outputs
- Write meaningful tests for LLM-powered applications

---

## Key Concepts

### LCEL and Chains
- LCEL: pipe operator `|`, `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`, `RunnableBranch`
- Output parsers: `StrOutputParser`, `PydanticOutputParser`, `JsonOutputParser`
- Structured output: `model.with_structured_output(MySchema)` — the reliable way to get typed data from any LLM
- Document loaders, text splitters, vector store integration (Chroma/FAISS for this phase)
- Chat history management: `MessagesPlaceholder`, `RunnableWithMessageHistory`
- Prompt management: LangSmith Hub — storing, versioning, and pulling prompts with `hub.pull("owner/prompt-name")`

### Agents and Tools
- `create_tool_calling_agent` + `AgentExecutor` — the standard agent pattern
- `@tool` decorator, `BaseTool` subclass, `StructuredTool.from_function()`
- Tool schemas: how JSON Schema drives what the model knows about a tool's inputs
- Memory patterns: `ConversationBufferMemory`, `ConversationSummaryMemory`, `ConversationTokenBufferMemory`

### Streaming
- `chain.stream()` and `chain.astream()` — synchronous and async streaming
- `chain.astream_events()` — fine-grained event stream: `on_chat_model_stream`, `on_tool_start`, `on_tool_end`
- Server-Sent Events (SSE) pattern: streaming LLM responses from a FastAPI endpoint to a browser client
- Backpressure and buffering considerations for streaming in production

### LangSmith
- Project setup, environment variables, automatic tracing with zero code changes
- Manual tracing: `@traceable` decorator, `with_` context manager
- Datasets and examples: building ground-truth QA pairs for evaluation
- Evaluators: `LLMAsJudge`, `ExactMatch`, custom Python evaluators
- LangSmith Hub for prompt version control

### Testing LLM Applications
- The challenge: LLM outputs are non-deterministic; tests need different strategies than unit tests
- **Deterministic tests**: verify the right tools were called, output is parseable, schemas validate
- **Regression tests**: golden dataset comparison — does the new prompt score at least as well as the baseline?
- **Semantic assertion tests**: embedding similarity threshold or LLM judge to assert output "means" the right thing
- `pytest` + `pytest-asyncio` for async LangChain code
- Mocking LLM calls: `FakeChatModel` from `langchain_core`
- `DeepEval` framework for structured LLM test assertions: https://docs.confident-ai.com/
- LangSmith evaluation runs as a CI step (run eval suite on PR, compare scores to baseline)

---

## Resources

- LangChain Python docs: https://python.langchain.com/docs/introduction/
- LCEL guide: https://python.langchain.com/docs/concepts/lcel/
- LangSmith docs: https://docs.smith.langchain.com/
- LangChain GitHub: https://github.com/langchain-ai/langchain
- DeepLearning.AI "LangChain for LLM Application Development" (free): https://learn.deeplearning.ai/courses/langchain
- DeepLearning.AI "LangChain: Chat with Your Data" (free): https://learn.deeplearning.ai/courses/langchain-chat-with-your-data
- Sam Witteveen YouTube (@samwitteveenai): practical LangChain deep dives
- DeepEval docs: https://docs.confident-ai.com/
- FastAPI streaming guide: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse

**Key paper:** ReAct (2022): https://arxiv.org/abs/2210.03629

---

## Hands-on Projects

1. **Document Q&A Chain** — Load PDFs, chunk, embed into Chroma, LCEL retrieval chain with custom prompt; prompt stored and versioned in LangSmith Hub; all traces visible in LangSmith
2. **Streaming Chat API** — FastAPI endpoint that streams LLM responses via SSE using `astream_events`; verify from a simple HTML client
3. **Structured Output Agent** — Agent with 3+ tools that always returns a validated Pydantic model; no raw string parsing allowed anywhere in the pipeline
4. **LLM Test Suite** — `pytest` suite with deterministic tests, golden dataset regression, and `DeepEval` semantic assertions for the Document Q&A chain

### Capstone: Research Assistant Agent
Tool-calling agent that searches multiple sources, synthesizes information, and streams a structured report to a FastAPI SSE endpoint. Fully traced in LangSmith. Test suite covers schema validation, tool invocation, and semantic quality regression.

---

## Completion Checklist

- [ ] LCEL chain with at least 3 runnables (`prompt | model | parser`) runs and traces in LangSmith
- [ ] FastAPI SSE endpoint streams tokens to a browser client in real time
- [ ] Agent calls the correct tool for at least 3 different query types
- [ ] `model.with_structured_output(MySchema)` returns a validated Pydantic model, not a string
- [ ] `pytest` suite passes: at least 1 deterministic test, 1 golden-set regression test, 1 DeepEval semantic test
- [ ] LangSmith Hub stores at least one versioned prompt used by the chain
- [ ] Capstone agent runs end-to-end: query → tool calls → structured report → SSE stream
