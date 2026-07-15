# Domain Adaptation Approach — Architectural Decision Record

## Context

Adapting a general-purpose LLM to a specific operational domain (e.g., OpenShift operations, platform SRE
runbooks) requires choosing between several complementary techniques. The domain corpus includes product
documentation, incident postmortems, runbooks, and operator logs. Evaluation was performed against a
50-question held-out benchmark covering factual recall, multi-hop reasoning, and procedural synthesis.
Metrics: answer accuracy (LLM-as-judge, GPT-4o), faithfulness (RAGAS), and answer relevancy (RAGAS).

## Options Evaluated

### 1. Naive RAG (Phase 3 baseline)

Standard dense retrieval with pgvector + cosine similarity. Chunks are embedded with
`text-embedding-3-small`. Retrieved top-k chunks are stuffed into a prompt template with no
re-ranking or graph context.

### 2. GraphRAG — Neo4j entity graph + Microsoft GraphRAG community reports

Documents are parsed into an entity–relation graph stored in Neo4j. Microsoft GraphRAG builds community
reports via hierarchical Leiden clustering. At query time, both local entity search (entity + neighbors)
and global community-report synthesis are available.

### 3. RAFT — Retrieval-Augmented Fine-Tuning

A base model (e.g., Mistral-7B or Llama-3-8B) is fine-tuned on domain documents using a mix of oracle
chunks (the document actually containing the answer) and distractor chunks (plausible-but-wrong context).
The model learns to identify and reason over relevant evidence, ignoring distractors. Fine-tuning uses
QLoRA via TRL/Axolotl on a single A100 or consumer GPU.

### 4. InstructLab — synthetic data generation via LAB training

A seed set of Q&A pairs and skill examples are expanded by a teacher model (e.g., Mixtral-8x7B or
GPT-4o) using the LAB (Large-scale Alignment for chatBots) methodology. The resulting synthetic dataset
trains a student model with knowledge and skill taxonomy updates. Compatible with the Red Hat InstructLab
CLI and the OpenShift AI model serving stack.

## Evaluation Table

| Approach | Strengths | Weaknesses | Best for |
|---|---|---|---|
| Naive RAG | Zero training cost; easy to update corpus; fast iteration | Struggles with multi-hop reasoning; hallucination on out-of-corpus questions; no thematic synthesis | Baselines, quick prototypes, low-stakes lookups |
| GraphRAG | Excels at global thematic queries ("what are the main failure modes?"); entity relationships captured explicitly | High indexing cost; slow community report generation; overkill for factual recall | Thematic synthesis, cross-document entity reasoning, exploration queries |
| RAFT | Strong factual accuracy on in-domain questions; learns to ignore distractors; RAG-at-inference not required | GPU + time for fine-tuning; model must be retrained when corpus changes significantly; inference infra needed | Stable, high-value corpora; latency-sensitive production endpoints |
| InstructLab | Community-friendly taxonomy contribution model; aligns model behavior and skills, not just knowledge; integrates well with RHEL AI | Slower iteration cycle; teacher model cost for synthetic data generation; less effective for purely factual dense retrieval | Behavioral alignment, skill acquisition, community-contributed domain knowledge |

## Decision

**Recommended architecture: RAFT + RAG as the primary approach for domains with high-quality document
corpora.**

- Use **RAFT** to fine-tune a base model on the domain corpus using oracle+distractor training. The
  fine-tuned model is served via vLLM on OpenShift AI and acts as the backbone.
- At inference time, pair the RAFT model with a **hybrid retrieval** pipeline (dense + BM25 + cross-encoder
  reranking) so the corpus can be updated without retraining.
- Use **InstructLab** for community-contributed knowledge and skills. Taxonomy updates flow into periodic
  retraining cycles rather than requiring full fine-tuning runs.
- Use **GraphRAG** selectively for queries classified as thematic/global by a lightweight query classifier
  (e.g., queries containing "what are the common patterns", "summarize all", "across documents").
- Keep **naive RAG** as the evaluation baseline and production fallback when the fine-tuned model is
  unavailable (e.g., during a rollout).

## Consequences

**What becomes easier:**
- Factual recall and multi-hop reasoning improve significantly over the naive RAG baseline.
- The hybrid retrieval layer means corpus updates (new runbooks, new operator docs) take effect immediately
  without retraining.
- InstructLab taxonomy contributions allow domain experts to improve the model without writing code.
- GraphRAG global search unlocks a class of thematic queries that dense retrieval cannot answer well.
- Defense-in-depth: if the fine-tuned model regresses, naive RAG remains functional.

**What becomes harder:**
- Operational complexity increases: RAFT fine-tuning pipeline, InstructLab retraining schedule, GraphRAG
  indexing job, and hybrid retrieval stack all need monitoring and maintenance.
- RAFT model must be periodically retrained as the corpus drifts; stale fine-tuning can hurt performance.
- GraphRAG community report indexing is expensive ($5–20 per 1 000 documents at API rates) and slow; it
  cannot be run on every corpus update.
- Debugging failures requires understanding which layer (retrieval, reranking, fine-tuned model weights,
  GraphRAG reports) introduced the error.
- InstructLab synthetic data quality depends on the teacher model and seed Q&A quality; poor seeds produce
  poor expansions.
