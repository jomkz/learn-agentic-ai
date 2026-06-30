# Phase 8: HuggingFace + LoRA/QLoRA + OpenShift AI + vLLM + KFP + Ray + MLOps

**Duration: 8 weeks** | [← Phase 7](07-llamastack.md) | [Phase 9 →](09-graphrag-raft-instructlab.md)

**Project directory:** [`projects/phase8-openshift/`](../../projects/phase8-openshift/)

> **Parallel track:** This phase can start alongside Phase 5 once Phase 3 is complete. The HuggingFace + LoRA weeks (1-3) are self-contained and don't require OpenShift access.

---

## Objectives

- Master the HuggingFace ecosystem as the foundation for all fine-tuning work
- Understand and apply LoRA/QLoRA for parameter-efficient fine-tuning
- Deploy and serve LLM models at scale with vLLM on OpenShift AI
- Build ML data and training pipelines with KubeFlow Pipelines v2
- Use Ray for distributed data processing and model serving
- Implement MLOps: model registry, monitoring, A/B testing
- Work with Podman for containerizing AI workloads (Red Hat tooling)

---

## Key Concepts

### Week 1-2: HuggingFace Ecosystem

The HuggingFace ecosystem is the foundation for all fine-tuning and model work in Phase 9. Learn it here before the complexity of RAFT and InstructLab.

- **`transformers`**: `AutoModelForCausalLM`, `AutoTokenizer`, `pipeline()`, `generate()` with sampling parameters
- **`datasets`**: loading from Hub and local files; streaming for large datasets; `.map()` for preprocessing; `DatasetDict` structure
- **HuggingFace Hub**: model cards, `push_to_hub()`, `hf_hub_download()`, private repos, `huggingface-cli login`
- **`tokenizers`**: fast tokenizers, batch encoding, padding/truncation, special tokens (`<|im_start|>`, `[INST]`)
- **`accelerate`**: device-agnostic training — `Accelerator`, mixed precision (fp16/bf16), gradient accumulation, `device_map="auto"` for multi-GPU
- **`PEFT`**: LoRA, QLoRA, adapters — the key library for Phase 9 fine-tuning
- **`trl`**: `SFTTrainer` for supervised fine-tuning; `DPOTrainer` for preference alignment; `DataCollatorForCompletionOnlyLM`
- **`evaluate`**: standardized metrics — BLEU, ROUGE, accuracy, perplexity
- Model formats: `safetensors` (HF default), converting to GGUF for Ollama, exporting to ONNX

### Week 2-3: LoRA and QLoRA

Understanding these techniques is required before RAFT in Phase 9.

**Why fine-tune at all?**
RAG handles factual knowledge retrieval. Fine-tuning handles: domain-specific reasoning style, output format requirements, consistent terminology, behavior that can't be prompted reliably.

**Full fine-tuning** — update all weights: requires many GPUs, high catastrophic forgetting risk. Rarely justified.

**LoRA (Low-Rank Adaptation)**
- Freeze the original model weights
- Add two small trainable matrices A (d×r) and B (r×d) next to each target layer
- During forward pass: output = original_weight @ x + (B @ A) @ x × (alpha/r)
- Key parameters:
  - `r`: rank (4-64); higher = more capacity but more memory; r=8 or r=16 common
  - `lora_alpha`: scaling factor; typically 2× rank
  - `target_modules`: which layers to adapt; usually `["q_proj", "v_proj", "k_proj", "o_proj"]`
  - `lora_dropout`: regularization; 0.05-0.1
- At inference: adapters can be merged into base weights or loaded separately (enables multiple adapter swapping)

**QLoRA**
- Quantize base model weights to 4-bit (NF4 format) with `bitsandbytes` — dramatically reduces VRAM
- Keep adapter computation in bf16/fp16
- A 7B model fits in ~6GB VRAM; a 13B model in ~10GB
- `BitsAndBytesConfig`: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`
- `prepare_model_for_kbit_training(model)` enables gradient checkpointing after quantization

**QLoRA training pipeline:**
```python
# 1. Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 2. Wrap with LoRA adapters
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, ...))

# 3. Train with SFTTrainer
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()

# 4. Save adapter weights (small — a few hundred MB)
model.save_pretrained("./my-adapter")

# 5. Optionally merge adapters into base weights for deployment
merged = model.merge_and_unload()
merged.save_pretrained("./my-merged-model")
```

### Week 3: Containerization with Podman

- Podman: Docker-compatible, daemonless, rootless — Red Hat standard
- `podman build`, `podman run`, `podman push` — same CLI as Docker; Dockerfiles work unchanged
- `podman-compose`: multi-container dev environments
- Building ML container images:
  - Base images: `pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime`, `registry.access.redhat.com/ubi9/python-311`
  - Multi-stage builds: large build stage, slim runtime stage
- Image registry: Quay.io (Red Hat) vs. Docker Hub

### Week 3-4: OpenShift AI / Open Data Hub

- Architecture: data science projects, workbenches (JupyterHub), model servers, pipelines, model registry
- ODH components: MLflow, Ray, KubeFlow Pipelines (Data Science Pipelines), ModelMesh, single-model servers
- OpenShift CLI: `oc` for OpenShift-specific resources (Routes, Projects); `kubectl` for standard Kubernetes resources
- Data Science Pipelines: OpenShift AI's managed KFP, integrated with the console UI

### Week 4: vLLM

- **PagedAttention**: non-contiguous KV cache blocks — key to high throughput (3-10× better than naive serving)
- **Continuous batching**: dynamically adds new requests to in-progress batches — eliminates head-of-line blocking
- Deployment on OpenShift: `Deployment` spec with `nvidia.com/gpu: 1` resource request, `Service`, `Route`
- OpenAI-compatible API: `/v1/chat/completions`, `/v1/embeddings` — zero code changes for any OpenAI SDK client
- Quantization: `--quantization gptq` or `--quantization awq` for reduced VRAM
- Multi-GPU: `--tensor-parallel-size 2` (or 4/8) for models exceeding single GPU VRAM
- Serving LoRA adapters: `--lora-modules adapter_name=/path/to/adapter` — one base model, multiple adapters
- vLLM + LlamaStack: configure a custom LlamaStack distribution pointing to the vLLM endpoint

### Week 5-6: KubeFlow Pipelines v2

- KFP v2 SDK: `@dsl.component` (Python function → container), `@dsl.pipeline` (workflow definition)
- Component I/O: typed artifacts — `Input[Dataset]`, `Output[Model]`, `Output[Artifact]`
- Lightweight Python components: run in a reusable base image; good for simple transformations
- Container components: full custom image; good for heavy dependencies
- Pipeline compilation: `compiler.Compiler().compile(pipeline_func, "pipeline.yaml")`
- Pipeline runs: submit via Python client or OpenShift AI console
- S3/MinIO artifact storage: artifacts flow between components via object store paths
- Caching: components cache outputs by default; skip unchanged upstream steps on reruns

### Week 6-7: Ray

- **Ray Core**: `@ray.remote` turns a function or class into a distributed task or actor; `ray.get()` blocks for result; `ray.put()` stores data in the distributed object store
- **Ray Data**: lazy, distributed data processing — `ray.data.read_parquet()`, `.map_batches(batch_fn, num_gpus=1)`, `.filter()`, `.write_parquet()`; processes datasets larger than single-node memory
- **Ray Train**: `TorchTrainer` with `ScalingConfig(num_workers=4, use_gpu=True)` for distributed PyTorch; checkpoint integration; works with HuggingFace `transformers`
- **Ray Serve**: `@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})` — scalable model serving with autoscaling policies
- **Ray on OpenShift**: KubeRay operator; `RayCluster` custom resource (head + worker nodes); `RayJob` for batch workloads

### Week 8: MLOps

- **MLflow**: `mlflow.start_run()` → log params, metrics, artifacts → `mlflow.register_model()` → stage transitions (None → Staging → Production → Archived)
- **Prometheus + Grafana on OpenShift**: `ServiceMonitor` CRD scrapes vLLM's `/metrics` endpoint; key vLLM metrics: `vllm:request_throughput`, `vllm:e2e_request_latency_seconds`, `vllm:gpu_cache_usage_perc`
- **A/B testing**: OpenShift Service Mesh (Istio) `VirtualService` traffic splitting (e.g., 80% model-v1, 20% model-v2); compare RAGAS scores and latency across variants
- **CI/CD for ML**: Tekton pipelines (OpenShift native) trigger KFP runs when new data arrives or code changes; quality gate in the pipeline promotes model version to Production in MLflow registry

---

## Resources

**HuggingFace**
- `transformers` docs: https://huggingface.co/docs/transformers
- `datasets` docs: https://huggingface.co/docs/datasets
- `PEFT` docs: https://huggingface.co/docs/peft
- `trl` docs: https://huggingface.co/docs/trl
- `accelerate` docs: https://huggingface.co/docs/accelerate
- DeepLearning.AI "Finetuning Large Language Models" (free): https://learn.deeplearning.ai/courses/finetuning-large-language-models
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314

**Podman**
- Podman docs: https://docs.podman.io/en/latest/
- Podman Compose: https://github.com/containers/podman-compose
- Red Hat UBI base images: https://catalog.redhat.com/software/containers/explore

**OpenShift AI / ODH**
- OpenShift AI docs: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed
- Open Data Hub GitHub: https://github.com/opendatahub-io

**vLLM**
- vLLM docs: https://docs.vllm.ai/en/latest/
- PagedAttention paper: https://arxiv.org/abs/2309.06180
- vLLM LoRA serving: https://docs.vllm.ai/en/latest/models/lora.html

**KubeFlow Pipelines**
- KFP v2 SDK docs: https://www.kubeflow.org/docs/components/pipelines/v2/

**Ray**
- Ray docs: https://docs.ray.io/en/latest/
- KubeRay operator: https://github.com/ray-project/kuberay

**MLflow / MLOps**
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Red Hat Developer YouTube: OpenShift AI demos

---

## Hands-on Projects

1. **QLoRA Fine-tuning** — Fine-tune Llama 3.2 3B on a small instruction dataset using QLoRA + TRL `SFTTrainer`; compare base vs. fine-tuned on 10 held-out prompts; push adapter to HuggingFace Hub
2. **Podman AI Service** — Containerize the Phase 3 RAG FastAPI service with Podman; multi-stage build; push to Quay.io; run with `podman-compose` (app + PostgreSQL)
3. **vLLM Deployment** — Deploy Llama 3.1 8B with vLLM on Kubernetes (Kind locally); test OpenAI-compatible API; benchmark throughput at 10/50/100 concurrent requests; serve the QLoRA adapter from Project 1 with `--lora-modules`
4. **KFP Pipeline** — 4-stage pipeline: data ingest → Docling parsing → embedding generation (Ray) → pgvector population; S3 artifacts between stages; run in OpenShift AI Data Science Pipelines
5. **Ray Distributed Processing** — Ray Data preprocessing over a 10K-document corpus; Ray Serve autoscaling embedding model; test autoscaling by ramping request rate
6. **MLflow + Prometheus** — Instrument the full RAG pipeline with MLflow tracking; Prometheus metrics on the FastAPI service; Grafana dashboard showing request rate, p95 latency, cache hit rate, and RAGAS score

### Capstone: Production RAG Platform on OpenShift AI
End-to-end system:
- Docling-based KFP pipeline (ingest → parse → embed → pgvector + Qdrant)
- vLLM serving the QLoRA fine-tuned Llama model
- Ray for distributed embedding generation
- LlamaStack distribution backed by the vLLM endpoint
- MLflow tracking with a quality gate promoting model to Production
- Prometheus + Grafana metrics dashboard
- A/B test: dense retrieval vs. hybrid retrieval on a live endpoint
- Fully containerized with Podman; deployment manifests for OpenShift AI

---

## Completion Checklist

- [ ] QLoRA fine-tuning runs to completion; adapter saved to HuggingFace Hub
- [ ] Fine-tuned model produces measurably better output on held-out prompts vs. base
- [ ] Podman image builds and runs; `podman-compose up` starts app + database together
- [ ] vLLM serves a model and responds to `curl` against `/v1/chat/completions`
- [ ] vLLM serves the LoRA adapter by name via `--lora-modules`
- [ ] KFP pipeline runs in OpenShift AI (or Kind) end-to-end; artifacts appear in S3/MinIO
- [ ] Ray Data processes 10K documents without OOM on a single node (distributed across workers)
- [ ] MLflow UI shows experiment runs with params, metrics, and registered model versions
- [ ] Grafana dashboard displays live vLLM throughput and p95 latency
- [ ] A/B test shows measurable quality or latency difference between retrieval strategies
