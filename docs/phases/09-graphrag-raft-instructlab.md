# Phase 9: GraphRAG, RAFT, InstructLab, and Multi-Modal Awareness

**Duration: 6 weeks** | [← Phase 8](08-huggingface-openshift.md) | [Capstone →](10-capstone.md)

**Project directory:** [`projects/phase9-domain-adaptive/`](../../projects/phase9-domain-adaptive/)

**Prerequisites:** Phases 4 (Advanced RAG) and 8 (LoRA/QLoRA, vLLM) must be complete before this phase.

---

## Objectives

- Implement GraphRAG for knowledge-graph-enriched retrieval
- Use RAFT to create domain-adapted fine-tuned models (builds directly on Phase 8 LoRA/QLoRA skills)
- Use InstructLab to contribute skills and knowledge to LLMs via synthetic data generation
- Gain practical awareness of multi-modal (vision-language) models and when they apply
- Know when each technique justifies its cost vs. vanilla RAG

---

## Key Concepts

### Week 1: Neo4j and Knowledge Graph Basics

A prerequisite for GraphRAG. You need to understand graph data models before the indexing pipeline makes sense.

**Graph data model vs. tables vs. documents:**
- Nodes: entities (Person, Document, Concept, Service)
- Relationships: typed, directed edges (`WROTE`, `REFERENCES`, `DEPENDS_ON`)
- Properties: key-value pairs on nodes and relationships

**Cypher query language essentials:**
```cypher
-- Create
CREATE (p:Person {name: "Alice"})-[:KNOWS]->(q:Person {name: "Bob"})

-- Match single hop
MATCH (p:Person)-[:KNOWS]->(friend) RETURN friend.name

-- Multi-hop traversal (find friends of friends)
MATCH (p:Person {name: "Alice"})-[:KNOWS*2]->(fof) RETURN fof.name

-- Aggregate
MATCH (d:Document)-[:HAS_ENTITY]->(e:Entity)
RETURN e.name, count(d) AS doc_count ORDER BY doc_count DESC LIMIT 10
```

**When knowledge graphs add value:**
- Entity-rich domains where relationships matter (biomedical, legal, product catalogs, software architecture)
- Multi-hop reasoning: "find all services that depend on ServiceA's downstream database"
- Global thematic summaries: "what are the main themes across 10,000 documents?"
- Queries that vector similarity alone can't answer

**Setup:**
```bash
podman-compose up -d neo4j
# UI: http://localhost:7474, Bolt: localhost:7687, Password: password
```

LangChain integration: `Neo4jGraph`, `GraphCypherQAChain`, `Neo4jVector`

### Week 2-3: GraphRAG (Microsoft)

**Why it improves on vector-only RAG:**
- Vector RAG excels at: "what does document X say about topic Y?" (local, specific retrieval)
- GraphRAG adds: "what are the main themes across the entire corpus?" (global, thematic synthesis) and multi-hop entity chains

**Microsoft GraphRAG indexing pipeline:**
1. **Entity extraction**: LLM reads every chunk, extracts named entities and the relationships between them (expensive — one LLM call per chunk)
2. **Graph construction**: entities become nodes, relationships become edges with text descriptions
3. **Community detection**: Leiden algorithm groups densely connected entity clusters into communities
4. **Community report generation**: LLM summarizes each community into a hierarchical report

**Search modes:**
- **Local search**: start from a specific entity, traverse neighbors, retrieve directly relevant context — best for entity-anchored questions
- **Global search**: query over community reports at multiple abstraction levels — best for broad thematic questions

**Cost reality check:**
- Indexing 1,000 documents may cost $5-20 in LLM API calls
- Use local Ollama models for indexing to control cost during learning
- Factor indexing cost into architecture decisions: GraphRAG makes sense when global search is a core use case, not for every project

**LightRAG** — simpler, faster alternative to Microsoft GraphRAG: https://github.com/HKUDS/LightRAG

**Decision framework:**
| Use case | Recommended approach |
|----------|---------------------|
| Direct fact lookup ("what is X?") | Naive RAG |
| Entity-anchored questions ("what does the contract say about party X?") | GraphRAG local search |
| Thematic synthesis ("what are the main risks across all contracts?") | GraphRAG global search |
| Multi-hop reasoning ("find services impacted by this database's deprecation") | GraphRAG |

### Week 3-4: RAFT (Retrieval-Augmented Fine-Tuning)

RAFT is a **fine-tuning recipe** built on top of the LoRA/QLoRA skills from Phase 8. It teaches a model how to reason over retrieved documents in your specific domain.

**Motivation:** Even with good retrieval, a general-purpose model may not know how to correctly reason over your domain's document format, terminology, or reasoning patterns. RAFT fixes this by training the model with domain documents.

**Training data construction:**
For each training question, construct a training example with:
- The question
- 1 oracle document (actually contains the answer) + K distractor documents (don't contain the answer, chosen randomly from the corpus)
- Target output: a chain-of-thought reasoning trace that quotes the oracle document, identifies the relevant passage, then gives the final answer
- 80% of examples include the oracle document; 20% omit it (teaches the model to answer from parametric memory when retrieval fails)

**Fine-tuning:** Use the Phase 8 pipeline — QLoRA + TRL `SFTTrainer`. The RAFT dataset is in chat format, same as any SFT dataset.

```bash
# Generate RAFT dataset (uses the RAFT script)
python raft.py \
  --datapath ./domain_docs/ \
  --output ./raft_dataset/ \
  --distractors 3 \
  --p 0.8 \
  --chunk_size 512 \
  --questions 5

# Fine-tune with QLoRA + TRL (same pipeline as Phase 8 Project 1)
```

**Evaluation:** Run RAGAS on (RAFT-tuned model + RAG) vs. (base model + same RAG). RAFT should show higher faithfulness — the model correctly cites retrieved passages instead of hallucinating.

### Week 4-5: InstructLab

InstructLab democratizes LLM instruction tuning — contributors add domain knowledge or skills without needing ML expertise, using a teacher model to generate synthetic training data.

**Taxonomy structure** (a Git repository of YAML files):
```
taxonomy/
├── knowledge/
│   └── my_domain/
│       └── qna.yaml      ← 5+ seed Q&A pairs + source document
└── compositional_skills/
    └── my_skill/
        └── qna.yaml      ← 5+ seed Q&A pairs
```

**Knowledge vs. Skills:**
- `knowledge/`: factual domain content that the model should know (e.g., your company's products, a technical domain)
- `compositional_skills/`: capabilities that generalize to new inputs (e.g., converting text to a specific format, a reasoning pattern)

**`ilab` CLI workflow:**
```bash
# 1. Initialize
ilab config init

# 2. Download a base model
ilab model download --repository instructlab/granite-7b-lab

# 3. Add your contribution to the taxonomy
# Edit taxonomy/knowledge/my_domain/qna.yaml

# 4. Generate synthetic training data (teacher model reads your seed Q&As and generates hundreds of variations)
ilab data generate --pipeline full

# 5. Train (use --num-epochs 1 for learning; real training needs more)
ilab model train --pipeline full --device cuda

# 6. Evaluate
ilab model evaluate --benchmark mt_bench

# 7. Serve and test
ilab model serve --model-path ./models/my-trained-model
ilab model chat
```

**LAB training phases:**
1. Knowledge training: learn the factual content from SDG-generated data
2. Skills training: learn the compositional skills
3. Replay: mix in general-purpose data to prevent catastrophic forgetting

**RHEL AI and OpenShift AI integration:** For large-scale training beyond a single GPU, use the InstructLab operator on OpenShift AI GPU nodes.

**Evaluation:** `ilab model evaluate` runs MT-Bench and an MMLU subset. For custom domain evaluation, use `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness

### Week 5-6: Production Monitoring and Drift Detection

Fine-tuned models and RAG pipelines degrade over time as data distributions shift. This is distinct from infrastructure metrics (latency, throughput — covered in Phase 8 with Prometheus/Grafana). Drift monitoring watches for *quality degradation* in the model's outputs and the data feeding it.

**Evidently AI** — open-source ML monitoring and data quality toolkit:
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TextEvals

# Data drift report — detect when retrieved chunks shift from training distribution
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df, column_mapping=ColumnMapping())
report.save_html("drift_report.html")

# LLM output quality monitoring
from evidently.descriptors import TextLength, Sentiment, SemanticSimilarity
llm_report = Report(metrics=[TextEvals(column_name="answer", descriptors=[
    Sentiment(),
    TextLength(),
    SemanticSimilarity(with_column="ground_truth"),
])])
```

**What to monitor in a RAG + fine-tuned model pipeline:**

| Signal | Tool | Alert condition |
|--------|------|----------------|
| Retrieved chunk embedding drift | Evidently `DataDriftPreset` on embeddings | Drift score > threshold → retrigger indexing |
| RAGAS scores over time | Evidently custom metric or MLflow | Faithfulness drops >5% week-over-week |
| Answer length / toxicity distribution | Evidently `TextEvals` | Sudden distribution shift |
| Input query distribution | Evidently `DataDriftPreset` on queries | Query topics shifting → may need new training data |
| Inference latency | Prometheus (Phase 8) | p95 > SLA threshold |

**Integration pattern — continuous evaluation loop:**
```
Production traffic (sampled) → log (query, retrieved_docs, answer) to S3/MinIO
    → nightly Evidently batch job (KFP component) → drift report → MLflow artifact
    → alert if RAGAS or drift score crosses threshold
    → trigger InstructLab/RAFT fine-tuning pipeline if quality gate fails
```

**Evidently Cloud vs. self-hosted:** Evidently is fully open-source; run the monitoring dashboard self-hosted alongside your existing Grafana/MLflow stack. The cloud offering is optional.

### Week 6: Multi-Modal Awareness

Practical overview of vision-language models — enough to make architectural decisions, not a full deep dive. Invest more here only if your documents are image-heavy.

**Vision-Language Models (VLMs):**
- Models that accept image + text input and produce text output
- Key models: Llama 3.2 Vision, Qwen-VL, PaliGemma, LLaVA
- Running locally: `ollama pull llama3.2-vision` — pass base64-encoded images in the message `content` array

**Multi-modal RAG pattern:**
1. Docling extracts images and tables from PDFs alongside text
2. VLM generates text descriptions for each image (captions, data extraction from charts)
3. Image descriptions are chunked and embedded alongside text chunks
4. At query time: retrieve from the combined index; LLM sees both text and image descriptions in context

**LangChain multi-modal message format:**
```python
HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
    {"type": "text", "text": "What does this chart show?"},
])
```

**When to invest further:** If your domain involves engineering drawings, financial charts, medical imaging, or other image-heavy documents, plan a dedicated sprint on multi-modal RAG. For mostly-text documents, this awareness is sufficient for architectural decisions.

---

## Resources

**Neo4j and Knowledge Graphs**
- Neo4j docs: https://neo4j.com/docs/
- Cypher manual: https://neo4j.com/docs/cypher-manual/current/
- Neo4j + LangChain: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

**GraphRAG**
- GraphRAG paper: https://arxiv.org/abs/2404.16130
- GraphRAG GitHub: https://github.com/microsoft/graphrag
- GraphRAG docs: https://microsoft.github.io/graphrag/
- LightRAG GitHub: https://github.com/HKUDS/LightRAG

**RAFT**
- RAFT paper: https://arxiv.org/abs/2403.10131
- RAFT GitHub: https://github.com/ShishirPatil/gorilla/tree/main/raft
- TRL `SFTTrainer` docs: https://huggingface.co/docs/trl/sft_trainer

**InstructLab**
- InstructLab docs: https://docs.instructlab.ai/
- InstructLab GitHub: https://github.com/instructlab/instructlab
- InstructLab taxonomy: https://github.com/instructlab/taxonomy
- LAB paper: https://arxiv.org/abs/2403.01081
- RHEL AI docs: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai
- `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness

**Monitoring and Drift Detection**
- Evidently AI docs: https://docs.evidentlyai.com/
- Evidently GitHub: https://github.com/evidentlyai/evidently
- Evidently LLM monitoring guide: https://docs.evidentlyai.com/user-guide/llm-evaluation

**Multi-Modal**
- Llama 3.2 Vision via Ollama: `ollama pull llama3.2-vision`
- Docling image extraction: https://ds4sd.github.io/docling/usage/
- LangChain multi-modal messages: https://python.langchain.com/docs/how_to/multimodal_inputs/

---

## Hands-on Projects

1. **Neo4j + Cypher Basics** — Model a small domain (e.g., software services: services, APIs, teams, dependencies) in Neo4j; write 10 Cypher queries covering CREATE, MATCH, multi-hop traversal, aggregation; query via LangChain `GraphCypherQAChain`

2. **GraphRAG vs. Naive RAG** — Run GraphRAG indexing pipeline on 500 domain articles (use Kubernetes docs or the same corpus from Phase 3); compare local search and global search quality to Phase 3 naive RAG using the RAGAS harness; measure indexing cost (tokens and time)

3. **RAFT Dataset + Fine-tuning** — Generate a RAFT training dataset for a domain using the RAFT script; fine-tune Llama 3.2 3B with QLoRA + TRL (Phase 8 pipeline); evaluate (RAFT-tuned model + RAG) vs. (base model + RAG) with RAGAS

4. **InstructLab Contribution** — Install `ilab`; write a knowledge contribution (≥5 Q&A pairs with a source document) for a domain you know; `ilab data generate`; inspect synthetic data quality and diversity; run abbreviated training (`--num-epochs 1`); evaluate with `ilab model evaluate`

5. **Evidently AI Monitoring Pipeline** — Log a sample of production queries, retrieved documents, and answers to a JSONL file over a week of evaluation runs; run Evidently `DataDriftPreset` comparing week-1 vs. week-2 embeddings; run `TextEvals` on answers for semantic similarity to ground truth; produce a drift report and set a threshold that would trigger retraining

6. **Multi-Modal Document Q&A** — Use Docling to extract images from a PDF with charts; pass extracted images to `llama3.2-vision` via Ollama for text descriptions; build a RAG pipeline that retrieves from both text chunks and image descriptions; compare to text-only retrieval on image-anchored questions

### Capstone: Domain-Adaptive Knowledge System

Choose one specific domain (e.g., OpenShift operations, automotive regulations, mobility telematics). Build and compare all four approaches on a 50-question held-out evaluation set:

| Approach | RAGAS score | Cost | Notes |
|----------|------------|------|-------|
| Naive RAG (Phase 3 baseline) | | | |
| GraphRAG (Neo4j + Microsoft GraphRAG) | | | |
| RAFT-tuned model + RAG | | | |
| InstructLab knowledge contribution | | | |

Deliverable: an **Architectural Decision Record (ADR)** in `projects/phase9-domain-adaptive/ADR.md` recommending when to use each approach, with supporting evaluation data from the comparison table.

---

## Completion Checklist

- [ ] Neo4j running via `podman-compose up neo4j`; `GraphCypherQAChain` answers a natural-language graph question
- [ ] GraphRAG indexing pipeline completes on 500 articles; both local and global search work
- [ ] GraphRAG global search answers a thematic question that naive RAG cannot answer well
- [ ] RAFT dataset generated with oracle + distractor documents and chain-of-thought labels
- [ ] RAFT-tuned model shows higher RAGAS faithfulness score than base model on the same retrieval pipeline
- [ ] `ilab data generate` produces ≥50 synthetic Q&A pairs from 5 seed examples
- [ ] Evidently drift report runs without errors on a sampled production log; at least one drift metric is above 0 (showing the tool is detecting signal, not noise)
- [ ] A retraining threshold is defined and documented: "if metric X exceeds Y, trigger pipeline Z"
- [ ] Multi-modal pipeline retrieves an image description chunk for an image-specific question
- [ ] Capstone ADR filled in with scores for all four approaches and a justified recommendation
