# Phase 4: Advanced RAG Techniques + Semantic Caching + Cost Optimization

**Duration: 3 weeks** | [← Phase 3](03-rag-fundamentals.md) | [Phase 5 →](05-langgraph.md)

**Project directory:** [`projects/phase4-advanced-rag/`](../../projects/phase4-advanced-rag/)
**Eval harness:** [`evals/`](../../evals/) — run the Phase 3 RAGAS harness against all techniques here

---

## Objectives

- Implement retrieval strategies that measurably outperform naive RAG
- Add semantic caching to reduce latency and LLM cost
- Build cost-optimization patterns essential for production AI architectures
- Establish evaluation-driven improvement loops

---

## Key Concepts

### Advanced Retrieval Techniques

- **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer, embed it for retrieval — improves semantic match for vague queries; the LLM never sees the answer, only the embedding is used
- **Multi-query retrieval**: generate N query reformulations, retrieve for each, fuse results with Reciprocal Rank Fusion (RAG-Fusion)
- **Contextual compression**: compress retrieved chunks down to only the relevant excerpt before passing to the LLM; reduces noise and tokens
- **Step-back prompting**: abstract the question before retrieving (e.g., "what is the K8s eviction threshold" → "what is Kubernetes resource management")
- **Parent-child chunking**: small chunks for retrieval precision; return surrounding large chunk for LLM context
- **Sentence window retrieval**: retrieve at sentence level, return N surrounding sentences
- **Cross-encoder reranking**: rescores top-k candidates using a dedicated model that reads query + document together
  - BGE-reranker (`BAAI/bge-reranker-large`): strong, local, free
  - Cohere Rerank API: best quality, API cost
  - FlashRank: fast, lightweight, local
- **Self-RAG**: model emits special tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`) to decide when to retrieve and critique its own output
- **CRAG** (Corrective RAG): assess retrieval relevance score; if below threshold, fall back to web search

### Semantic Caching
- Why cache: LLM inference is expensive and slow; many real-world queries are semantically similar
- **Redis Semantic Cache** (`langchain_community.cache.RedisSemanticCache`): embed the query, check vector similarity in Redis; return cached response if score exceeds threshold
- Cache hit rate vs. staleness tradeoffs: TTL configuration, invalidation on document updates
- Cache warming: pre-populate with known frequent queries
- When NOT to cache: highly personalized responses, real-time data queries, low-repetition workloads

### LLM Cost Optimization Patterns
- **Model tiering**: cheap/fast model (Llama 3.2 3B, GPT-4o-mini, Claude Haiku) for routing, classification, and summarization; expensive model only for final synthesis
- **Prompt caching**: Anthropic and OpenAI support prompt prefix caching — cache the system prompt + documents prefix; can reduce cost 80-90% for repeated RAG calls
- **Token budgeting**: count tokens before calling LLM (`tiktoken`, HuggingFace `AutoTokenizer`); truncate context dynamically to stay under budget
- **Batching**: group multiple embedding requests; use `asyncio.gather` for concurrent LLM calls
- **Streaming for perceived performance**: start showing results immediately even if total time is unchanged

### Context Window Management
- Stuffing: put all retrieved context in one call — simple but hits limits
- Map-reduce: process chunks independently, then reduce — good for summarization
- Refine: iteratively refine answer over each chunk — good for sequential reasoning
- Summarize-then-query: compress long documents before adding to context

### DSPy — Declarative, Optimizable Prompting

DSPy (Stanford) replaces hand-crafted prompt strings with **typed signatures and learnable modules**. Instead of writing "Answer the question given the context", you declare what goes in and what comes out, then DSPy compiles + optimizes the prompt automatically against a metric.

**Why this matters for RAG architectures:**
- Hand-crafted prompts are brittle: changing the model or data distribution breaks them silently
- DSPy treats prompts as a program to optimize, not a string to maintain
- Enables systematic, metric-driven prompt improvement — the same workflow as hyperparameter tuning

**Core primitives:**
```python
import dspy

# 1. Define a typed signature — the contract, not the prompt
class RAGSignature(dspy.Signature):
    """Answer a question given retrieved context passages."""
    context: list[str] = dspy.InputField(desc="retrieved passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="concise, faithful answer")

# 2. Use a module — DSPy generates and manages the prompt
rag_module = dspy.ChainOfThought(RAGSignature)

# 3. Define a metric — what "good" means
def faithfulness_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# 4. Optimize — DSPy tries different prompt strategies to maximize the metric
optimizer = dspy.BootstrapFewShot(metric=faithfulness_metric)
optimized_rag = optimizer.compile(rag_module, trainset=train_examples)
```

**Key optimizers:**
- `BootstrapFewShot`: automatically selects and labels few-shot examples
- `MIPROv2`: full prompt optimization — instructions + examples; requires more data
- `BetterTogether`: combines retrieval and generation optimization

**When to use DSPy vs. hand-crafted prompts:**
- Use DSPy when you have a measurable metric and a labeled eval set (which you do — RAGAS)
- Use hand-crafted prompts for one-off tasks with no systematic evaluation loop
- DSPy shines in production RAG where prompt quality needs to be maintained over time

**DSPy + LangChain/LangGraph:** DSPy handles individual module optimization; LangGraph handles stateful orchestration. They compose naturally — use DSPy `Predict`/`ChainOfThought` inside LangGraph nodes.

---

## Resources

- HyDE paper: https://arxiv.org/abs/2212.10496
- Self-RAG paper: https://arxiv.org/abs/2310.11511
- CRAG paper: https://arxiv.org/abs/2401.15884
- RAG Survey: https://arxiv.org/abs/2312.10997
- Lance Martin's "RAG from Scratch": https://github.com/langchain-ai/rag-from-scratch
- DeepLearning.AI "Building and Evaluating Advanced RAG": https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag
- Redis semantic cache (LangChain): https://python.langchain.com/docs/integrations/llm_caching/
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- tiktoken: https://github.com/openai/tiktoken
- DSPy docs: https://dspy.ai/
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- DSPy paper (2023): https://arxiv.org/abs/2310.03714
- DSPy + RAG tutorial: https://dspy.ai/tutorials/rag/

---

## Hands-on Projects

1. **Technique Benchmarking** — HyDE vs Multi-Query vs Reranking on the Phase 3 RAGAS harness; compare quality score, latency (ms/query), and estimated cost (tokens used)
2. **Reranker Integration** — BGE-reranker (local) + Cohere Rerank (API); measure precision@3 and precision@5 improvement; note latency cost of reranking
3. **Semantic Cache** — Add Redis semantic cache to the Phase 3 documentation search; run a realistic query workload; report cache hit rate and cost savings estimate
4. **Cost-Optimized Architecture** — Refactor the Phase 2 Research Agent to use model tiering: cheap model for tool routing decisions, expensive model only for final answer synthesis; measure cost reduction percentage

5. **DSPy RAG Optimizer** — Take the Phase 3 RAGAS harness and 50 labeled QA pairs; define a `RAGSignature`; compile with `BootstrapFewShot` using faithfulness as the metric; compare RAGAS scores before and after optimization; inspect the auto-generated few-shot examples DSPy chose

---

## Completion Checklist

- [ ] HyDE, Multi-Query, and Reranking each have RAGAS scores recorded in a comparison table
- [ ] At least one technique improves RAGAS faithfulness or context precision vs. Phase 3 naive baseline
- [ ] Redis semantic cache is running (`podman-compose up redis`) and returning cached results on repeated queries
- [ ] Cache hit rate is measurable (instrumented with logging or metrics)
- [ ] Model tiering reduces token cost by ≥30% vs. single-model approach on the same task
- [ ] Token budget enforcer prevents any LLM call from exceeding a configured limit
- [ ] DSPy `BootstrapFewShot` optimizer improves RAGAS faithfulness score vs. hand-crafted prompt baseline
- [ ] DSPy auto-generated few-shot examples are inspectable and make sense for the domain
