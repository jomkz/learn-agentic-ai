# RAGAS Evaluation: Retrieval Strategy Comparison

This report compares five retrieval strategies on the held-out evaluation set. Run the
evaluation harness to populate the TBD cells before using this table for decision-making.

## How to Run

```bash
# Run all strategies against the held-out eval set
uv run python evals/ragas_harness.py \
    --eval-set evals/data/capstone_eval_set.jsonl \
    --strategies all \
    --output evaluation/ragas_results.json

# Run a single strategy
uv run python evals/ragas_harness.py \
    --eval-set evals/data/capstone_eval_set.jsonl \
    --strategies naive_rag \
    --output evaluation/ragas_results.json
```

Replace each `TBD` cell below with the value from `ragas_results.json` for the corresponding
strategy and metric. Rerun after any change to the retrieval pipeline or model.

---

## Results Table

| Strategy | Phase | Faithfulness | Answer Relevancy | Context Precision | Latency p95 (s) | Cost / Query ($) |
|----------|-------|-------------|-----------------|-------------------|-----------------|-----------------|
| Naive RAG | 3 | TBD — run: `uv run python evals/ragas_harness.py` | TBD | TBD | TBD | TBD |
| + Reranking | 4 | TBD — run: `uv run python evals/ragas_harness.py` | TBD | TBD | TBD | TBD |
| + Hybrid Search | 3 | TBD — run: `uv run python evals/ragas_harness.py` | TBD | TBD | TBD | TBD |
| + GraphRAG | 9 | TBD — run: `uv run python evals/ragas_harness.py` | TBD | TBD | TBD | TBD |
| + RAFT-tuned | 9 | TBD — run: `uv run python evals/ragas_harness.py` | TBD | TBD | TBD | TBD |

---

## Metric Definitions

### Faithfulness

Range: 0.0 – 1.0 (higher is better).

Measures whether every claim in the generated answer is supported by the retrieved context.
RAGAS decomposes the answer into atomic statements and checks each one against the context
using an LLM judge. A score of 1.0 means every statement is grounded; 0.0 means none are.

**Target**: >= 0.85. Scores below 0.75 indicate the model is hallucinating facts not present
in the retrieved chunks. Common causes: insufficient reranking, too few retrieved chunks, or
a model that has been fine-tuned to sound authoritative regardless of context.

### Answer Relevancy

Range: 0.0 – 1.0 (higher is better).

Measures whether the generated answer addresses the user's question. RAGAS generates several
hypothetical questions from the answer and measures their semantic similarity to the original
question. High faithfulness with low answer relevancy means the answer is grounded but off-
topic (the retrieved chunks were irrelevant to the query).

**Target**: >= 0.80. Low scores often point to retrieval failures (wrong chunks retrieved)
rather than generation failures.

### Context Precision

Range: 0.0 – 1.0 (higher is better).

Measures whether the retrieved chunks are ranked with the most relevant ones first. Specifically,
it checks whether the chunks that are actually used in the answer (as determined by the
faithfulness judge) appear early in the ranked list. High context precision means reranking is
working; low context precision means relevant chunks are buried.

**Target**: >= 0.75. The jump from naive RAG to + Reranking should be the largest improvement
on this metric. If it is not, the cross-encoder reranker may not be calibrated for this domain.

### Latency p95 (seconds)

The 95th-percentile end-to-end query latency in seconds, measured from query receipt to first
SSE token. Includes: cache miss check, retrieval, reranking, and LLM first-token latency.
Cache hits are excluded from this measurement (they are tracked separately as cache hit rate).

**Target**: <= 3.0 seconds. Strategies that add graph traversal (GraphRAG) or a second retrieval
pass (RAFT-tuned with verification) will have higher latency. The latency/quality trade-off must
be acceptable for the target user experience.

### Cost per Query ($)

Estimated cost in USD per query, including LLM input tokens (retrieved context + system prompt +
query) and output tokens (answer). Does not include embedding cost (amortised over corpus size)
or retrieval infrastructure cost. Calculated using the vLLM token count multiplied by the
per-token price of the model being served.

**Target**: <= $0.01 per query for the production deployment at scale. Strategies with larger
context windows (GraphRAG, RAFT-tuned with longer prompts) will have higher token counts and
therefore higher cost per query.

---

## How to Read the Table

1. **Baseline (Naive RAG)**: establishes the floor. All other strategies should improve on at
   least one metric without regressing others.

2. **+Reranking**: expect the largest improvement in context precision. If faithfulness also
   improves significantly, the naive RAG retrieval was noisy (irrelevant chunks in context).

3. **+Hybrid Search**: expect improvement in answer relevancy for queries with specific named
   entities or rare terminology, where dense retrieval alone misses relevant chunks.

4. **+GraphRAG**: expect improvement on multi-hop reasoning questions that require connecting
   information across documents. Expect higher latency due to graph traversal. If faithfulness
   does not improve on this eval set, the eval set may not contain multi-hop questions — check
   the eval set composition before drawing conclusions.

5. **+RAFT-tuned**: expect improvement in faithfulness on domain-specific questions. If the
   RAFT adapter was trained on a different domain than the eval set, this row may underperform
   naive RAG — the adapter is domain-specific, not universal.

---

## Notes on the Eval Set

The eval set (`evals/data/capstone_eval_set.jsonl`) should contain at minimum 100 question-
context-answer triples drawn from the target document corpus. Aim for a distribution of:
- 40% single-hop factual questions
- 30% multi-hop reasoning questions
- 20% questions with named entity or terminology specificity
- 10% adversarial questions (ambiguous, out-of-corpus, or trick questions)

A skewed eval set will produce misleading metric rankings. Validate the eval set composition
before presenting results to stakeholders.
