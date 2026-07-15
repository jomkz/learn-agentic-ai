# Enterprise Knowledge Assistant Platform — Operational Runbook

This runbook covers day-to-day operations: starting the local development environment, running
pipelines, updating models, and responding to alerts.

---

## Local Development Environment

### Start all services

```bash
podman-compose up -d
```

### Start only the services needed for a specific phase

```bash
# Phase 3+ (RAG, pgvector)
podman-compose up -d postgres qdrant

# Phase 4+ (semantic cache)
podman-compose up -d redis

# Phase 9 (GraphRAG)
podman-compose up -d neo4j

# Everything
podman-compose up -d
```

### Stop services

```bash
podman-compose down           # stop, keep volume data
podman-compose down -v        # stop and delete all volumes (clean slate)
```

### Service URLs

| Service | URL | Default credentials |
|---------|-----|---------------------|
| PostgreSQL (pgvector) | `localhost:5432` | See `.env` |
| Qdrant REST API | `http://localhost:6333` | None (local) |
| Qdrant gRPC | `localhost:6334` | None (local) |
| Redis | `localhost:6379` | See `.env` |
| Neo4j Browser | `http://localhost:7474` | neo4j / see `.env` |
| Neo4j Bolt | `bolt://localhost:7687` | neo4j / see `.env` |
| MLflow UI | `http://localhost:5000` | None (local) |
| Jupyter Lab | `http://localhost:8888` | Token in terminal output |

### Copy and configure environment

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, HF_TOKEN, LANGCHAIN_API_KEY, etc.
```

---

## Running the KFP Ingestion Pipeline

The ingestion pipeline parses, chunks, embeds, and stores new documents. Run it any time the
document corpus changes.

### Trigger via KFP UI (production on OpenShift AI)

1. Open the KFP UI at the route exposed by the OpenShift AI operator.
2. Navigate to Pipelines > `document-ingestion-pipeline`.
3. Click "Create run". Set the parameter `corpus_bucket` to the MinIO bucket name.
4. Click "Start". Monitor run progress in the KFP UI.

### Trigger via CLI (local dev or CI)

```bash
uv run python projects/capstone/pipelines/run_ingestion.py \
    --corpus-bucket my-corpus-bucket \
    --endpoint http://localhost:8888  # KFP endpoint; use OpenShift route in prod
```

### Monitor pipeline progress

```bash
# Tail logs for the most recent run
uv run python projects/capstone/pipelines/tail_run_logs.py --run-id <run-id>
```

### Verify ingestion success

```bash
# Check chunk count in pgvector
uv run python -c "
from projects.capstone.db import get_chunk_count
print('Chunks in pgvector:', get_chunk_count())
"

# Check collection size in Qdrant
curl http://localhost:6333/collections/enterprise-docs | python -m json.tool
```

---

## Updating the Domain-Adapted Model

The platform uses LoRA adapters for domain adaptation. The update workflow is:

### Step 1: Generate or update the training dataset

```bash
uv run python evals/raft_dataset_generator.py \
    --corpus-bucket new-domain-bucket \
    --output data/raft_training/new_domain.jsonl
dvc add data/raft_training/new_domain.jsonl
git add data/raft_training/new_domain.jsonl.dvc && git commit -m "add new domain training data"
```

### Step 2: Train the adapter with InstructLab or RAFT

```bash
# RAFT fine-tuning
uv run python projects/capstone/training/train_raft.py \
    --dataset data/raft_training/new_domain.jsonl \
    --output-dir adapters/raft-new-domain \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --lora-rank 16 \
    --epochs 3

# Log to MLflow
uv run python projects/capstone/training/log_adapter.py \
    --adapter-path adapters/raft-new-domain \
    --experiment-name raft-adapters \
    --run-name new-domain-v1
```

### Step 3: Evaluate the adapter

```bash
uv run python evals/ragas_harness.py \
    --eval-set evals/data/capstone_eval_set.jsonl \
    --adapter-path adapters/raft-new-domain \
    --output evaluation/ragas_results_new_domain.json
```

The adapter must achieve `faithfulness >= 0.85` and must not regress `answer_relevancy` by
more than 0.05 compared to the current production adapter before promotion.

### Step 4: Push adapter to HuggingFace Hub

```bash
uv run python projects/capstone/training/push_adapter.py \
    --adapter-path adapters/raft-new-domain \
    --hub-repo your-org/raft-new-domain-adapter
```

### Step 5: Register adapter in MLflow and promote to production

```bash
uv run python projects/capstone/training/promote_adapter.py \
    --mlflow-run-id <run-id-from-step-2> \
    --stage production
```

### Step 6: Hot-swap the adapter in vLLM

vLLM supports hot-swapping LoRA adapters without restarting the server. Update the Deployment
to point to the new adapter path, or use the vLLM model management API:

```bash
# Update the Kubernetes Deployment (triggers a rolling restart)
kubectl set env deployment/vllm-server -n ai-platform \
    LORA_MODULE_PATH=/adapters/raft-new-domain

# Or use vLLM's live adapter API (no restart required)
curl -X POST http://vllm-server:8000/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "raft-adapter", "lora_path": "/adapters/raft-new-domain"}'
```

---

## Adding a New Document Corpus

To index a new set of documents:

1. Upload documents to the MinIO corpus bucket:

```bash
mc cp --recursive /path/to/new-docs/ minio/my-corpus-bucket/new-docs/
```

2. Trigger the KFP ingestion pipeline with the new bucket (see "Running the KFP Ingestion
   Pipeline" above).

3. Verify chunk count increases in both pgvector and Qdrant.

4. If the new corpus is from a new domain, consider training a domain-specific RAFT adapter
   (see "Updating the Domain-Adapted Model" above).

---

## Rolling Back a Bad Model Version in MLflow

If a new adapter degrades production quality:

### Option A: Promote the previous adapter version back to production

```bash
# List adapter versions
uv run python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
for mv in client.search_model_versions(\"name='raft-adapter'\"):
    print(mv.version, mv.current_stage, mv.run_id)
"

# Promote a previous version back to production
uv run python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name='raft-adapter',
    version='<previous-version-number>',
    stage='Production',
    archive_existing_versions=True
)
"
```

### Option B: Revert the Kubernetes Deployment to the previous adapter

```bash
kubectl rollout undo deployment/vllm-server -n ai-platform
kubectl rollout status deployment/vllm-server -n ai-platform
```

### Option C: Disable the adapter entirely (fall back to base model)

```bash
kubectl set env deployment/vllm-server -n ai-platform LORA_MODULE_PATH=""
```

After any rollback, run the RAGAS evaluation harness to confirm quality has been restored before
notifying users.

---

## Reading the Grafana Dashboard

Open the Grafana dashboard at the route exposed in the `monitoring` namespace. The primary
dashboard is "Enterprise Knowledge Assistant Platform".

### Key metrics

**vllm:request_throughput** (requests/second)
- Normal range: 1–10 req/s depending on query complexity
- Alert threshold: drops below 0.5 req/s for more than 2 minutes (potential deadlock or OOM)
- Panel: "Inference Throughput" (top left)

**vllm:e2e_request_latency_seconds** (histogram, track p50/p95/p99)
- Normal p95: 2.0–3.5 seconds for cache-miss queries
- Normal p50: 1.0–2.0 seconds
- Alert threshold: p95 > 5.0 seconds for more than 5 minutes
- Panel: "Inference Latency" (top right)
- Note: a sudden p95 spike without a throughput drop usually indicates a single slow request
  (long document context); check the LangSmith trace for the outlier.

**cache_hit_rate** (ratio 0.0–1.0)
- Normal range: 0.2–0.6 for typical enterprise helpdesk workloads
- A drop to 0.0 may indicate Redis is down or the semantic similarity threshold is too strict
- A sustained rate above 0.8 may indicate users are asking the same questions (investigate
  whether the canonical answers are correct)
- Panel: "Semantic Cache" (middle left)

**GPU utilisation** (percent)
- Normal range during active load: 60–90%
- Below 20% during business hours: vLLM may be idle due to a front-end issue (check the
  LangGraph application logs)
- Sustained 100%: scale the vLLM Deployment replicas or increase GPU allocation
- Panel: "GPU Utilisation" (middle right)

**Error rate** (ratio of 5xx responses)
- Alert threshold: > 1% for more than 2 minutes
- Panel: "Error Rate" (bottom left)

---

## Alert Response Procedures

### Alert: p95 latency > 5 seconds

1. Check GPU utilisation. If > 95%, the GPU is saturated: scale the Deployment or reduce
   concurrent requests.
2. Check the cache hit rate. If it has dropped to near 0, Redis may be down: `podman-compose
   ps redis` (local) or `kubectl get pod -n ai-platform -l app=redis` (prod).
3. Check LangSmith for the slow trace. If a single trace is dominating, it may be a very long
   document context: check the chunk count in the retrieval result.
4. If the vLLM pod is OOMKilled: increase the memory limit in `deploy/vllm-deployment.yaml`
   and redeploy.

### Alert: error rate > 1%

1. Check the LangGraph application logs: `kubectl logs -n ai-platform -l app=langgraph-app
   --tail=100`.
2. Check the vLLM logs: `kubectl logs -n ai-platform -l app=vllm-server --tail=100`.
3. If errors are 503s from vLLM: the server may be restarting (model load). Check
   `kubectl describe pod -n ai-platform -l app=vllm-server` for recent events.
4. If errors are LlamaGuard classification failures: check that the LlamaGuard vLLM instance
   (small model) is running. It runs on a separate Deployment (`vllm-server-small`).

### Alert: pipeline failure (KFP)

1. Open the KFP UI and navigate to the failed run.
2. Click on the failed node to see its logs.
3. Common causes: MinIO connectivity (check credentials in `.env`), OOM during embedding
   (reduce Ray batch size in `projects/capstone/pipelines/embed_stage.py`), pgvector
   connection pool exhaustion (check `max_connections` in PostgreSQL config).
4. Re-trigger the pipeline after fixing the root cause. The pipeline is idempotent: re-running
   it will overwrite existing vectors for re-processed documents.

### Alert: Qdrant collection unhealthy

```bash
# Check collection status
curl http://localhost:6333/collections/enterprise-docs

# Rebuild collection index (takes several minutes for large collections)
curl -X POST http://localhost:6333/collections/enterprise-docs/index
```

### Escalation path

If an alert cannot be resolved within 30 minutes using the steps above:
1. Capture the LangSmith trace ID, Grafana screenshot, and relevant pod logs.
2. File an incident in the team issue tracker with the `P1-incident` label.
3. Notify the on-call ML platform engineer via the team incident channel.
