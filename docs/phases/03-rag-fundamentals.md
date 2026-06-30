# Phase 3: RAG Fundamentals — Document Parsing, Embeddings, Vector Stores, Retrieval

**Duration: 4 weeks** | [← Phase 2](02-langchain.md) | [Phase 4 →](04-advanced-rag.md)

**Project directory:** [`projects/phase3-rag/`](../../projects/phase3-rag/)
**Eval harness output:** [`evals/`](../../evals/) ← the RAGAS harness built here is reused in all later phases

---

## Objectives

- Parse complex real-world documents (PDFs with tables, images, mixed formats)
- Understand embeddings, similarity metrics, and chunking tradeoffs deeply
- Work with production-grade vector databases: pgvector, Milvus, Qdrant, Weaviate
- Understand where LlamaIndex fits alongside LangChain for RAG workloads
- Build a systematic RAGAS evaluation pipeline reusable across all future phases

---

## Key Concepts

### Document Parsing — The Underrated First Step
Poor parsing kills RAG quality before retrieval even begins.

- **Docling** (Red Hat open-source): intelligent PDF/DOCX/HTML parsing preserving tables, headings, lists, and image captions; outputs structured JSON or Markdown
  - `DoclingLoader` integrates directly with LangChain
  - Handles complex PDFs that PyPDF/pdfplumber fail on (multi-column, scanned, tables)
- `Unstructured.io`: general-purpose document parsing for diverse formats
- `pymupdf4llm`: fast, lightweight PDF-to-Markdown conversion
- When to use each: Docling for structure-critical enterprise docs; pymupdf4llm for speed; Unstructured for format variety
- Image extraction and multi-modal document processing (brief intro — deep coverage in Phase 9)

### Embeddings
- Dense models: BGE (`BAAI/bge-large-en-v1.5`), E5 (`intfloat/e5-large-v2`), Sentence Transformers (`all-mpnet-base-v2`), OpenAI `text-embedding-3-small/large`
- Sparse models: BM25 (term frequency), SPLADE
- Similarity metrics: cosine (normalized vectors), dot product (unnormalized), L2 — when each applies
- Running embedding models locally with Ollama: `ollama pull nomic-embed-text`

### Chunking Strategies
- Fixed-size with overlap: simple, predictable, loses structure
- Recursive character: LangChain's default; respects paragraph/sentence boundaries
- Semantic chunking: split on embedding similarity drops between sentences
- Document-structure-aware: chunk by heading, section, or element type (Docling enables this)
- Chunk overlap and its effect on retrieval boundary cases
- Metadata enrichment: attach source, section, page number to every chunk

### Vector Stores

| Store | Best for | Notes |
|-------|----------|-------|
| **pgvector** | Production default; SQL co-location | Runs in existing PostgreSQL; HNSW indexing |
| **Qdrant** | High-scale production; rich filtering | Rust-based; excellent payload filtering; cloud or self-hosted |
| **Milvus** | Distributed, very large scale | Kubernetes-native; more ops overhead |
| **Weaviate** | Schema-first; built-in vectorization | GraphQL API; good for structured data |
| **Chroma** | Local dev and prototyping | Zero-ops; not for production |
| **FAISS** | Research and offline batch search | In-memory/file; no serving layer |

- Hybrid search: dense (semantic) + sparse (BM25) combined with Reciprocal Rank Fusion (RRF)
- Metadata filtering: pre-filter by source, date, category before vector similarity ranking
- Vector store internals: HNSW (graph-based, approximate), IVF (inverted file, cluster-based), flat (exact)

### LlamaIndex vs. LangChain for RAG
LlamaIndex is the other major RAG framework — know both:

- **LlamaIndex strengths**: richer built-in index types (`VectorStoreIndex`, `KnowledgeGraphIndex`, `SummaryIndex`), deeper document hierarchy modeling, excellent for complex structured retrieval
- **LangChain strengths**: broader ecosystem (agents, tools, chains), better for multi-step pipelines beyond retrieval
- **In practice**: many production systems use LangChain for orchestration + LlamaIndex concepts for retrieval design, or pick one and stay consistent
- Key LlamaIndex primitives: `VectorStoreIndex`, `QueryEngine`, `RetrieverQueryEngine`, `NodeParser`

### Evaluation with RAGAS
- Metrics: faithfulness, answer relevancy, context precision, context recall, answer correctness
- Dataset structure: (question, ground-truth-answer, retrieved-context) triples
- Automated evaluation with LLM judges vs. human evaluation — cost/quality tradeoffs
- Build the harness once here; reuse across Phases 4-9

---

## Resources

- Docling GitHub: https://github.com/DS4SD/docling | Docs: https://ds4sd.github.io/docling/
- Unstructured.io docs: https://docs.unstructured.io/
- pgvector GitHub: https://github.com/pgvector/pgvector
- Qdrant docs: https://qdrant.tech/documentation/
- Milvus docs: https://milvus.io/docs
- Weaviate docs: https://weaviate.io/developers/weaviate
- Chroma docs: https://docs.trychroma.com/
- LlamaIndex docs: https://docs.llamaindex.ai/en/stable/
- DeepLearning.AI "Building and Evaluating Advanced RAG" (free): https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag
- DeepLearning.AI "Building Agentic RAG with LlamaIndex" (free): https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex
- Greg Kamradt's chunking tutorials: https://github.com/FullStackRetrieval-com/RetrievalTutorials

**Key papers:** RAGAS (2023): https://arxiv.org/abs/2309.15217 | Sentence-BERT: https://arxiv.org/abs/1908.10084

---

## Hands-on Projects

1. **Document Parsing Comparison** — Take a complex multi-column PDF with tables; parse with Docling, pymupdf4llm, and PyPDF; measure table extraction quality, heading preservation, and chunk coherence
2. **Embedding Model Comparison** — Same corpus, 3 embedding models (BGE, E5, OpenAI `text-embedding-3-small`); measure retrieval quality on a hand-curated QA set with RAGAS
3. **Chunking Experiment** — Same Docling-parsed document, 4 chunking strategies; measure RAGAS score impact; identify which strategy wins for which document type
4. **pgvector + Qdrant Setup** — Same dataset in both; compare query latency, metadata filtering, and operational complexity
5. **RAGAS Evaluation Harness** — Reusable pipeline in `evals/` that takes any (retriever, dataset) pair and produces a RAGAS scorecard

### Capstone: Technical Documentation Search
Production RAG over Kubernetes docs. Docling for parsing. pgvector + hybrid search. RAGAS evaluation. FastAPI service with SSE streaming (from Phase 2). Deployed locally with Podman Compose.

---

## Completion Checklist

- [ ] Docling successfully parses a complex multi-column PDF and preserves table structure
- [ ] RAGAS harness in `evals/` runs against any retriever and produces a score report
- [ ] pgvector running via `podman-compose up postgres` with HNSW index created
- [ ] Qdrant running via `podman-compose up qdrant` with a collection and payload filter tested
- [ ] Embedding comparison shows measurable RAGAS score difference across 3 models
- [ ] Hybrid search (dense + BM25) outperforms pure dense search on at least one query type
- [ ] Capstone FastAPI service answers Kubernetes questions with sources, streams responses
