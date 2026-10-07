# Deployment exercises — cluster verification pending

The YAML files in this directory are design templates, not a complete installation. They
require cluster operators, credentials, GPU nodes, model/adapter artifacts, PVCs, and services
not provisioned here. No OpenShift deployment or throughput claim is verified by unit tests.

The old hand-written KFP spec referenced nonexistent modules and has been removed. Compile
from the executable CLI components instead:

```bash
uv sync --locked --extra pipelines
podman build -f Containerfile -t YOUR_REGISTRY/mobility-ai:YOUR_TAG .
# Push to your registry before submitting the pipeline to a cluster.
uv run python -m mobility_ai.phase8.kfp_pipeline \
  --image YOUR_REGISTRY/mobility-ai:YOUR_TAG --output outputs/kfp-pipeline.yaml
```

Supply `corpus_uri` and `questions_uri` as accessible KFP artifact URIs. The corpus artifact
must contain UTF-8 `.txt` or `.md` files; questions are JSONL. `provider=lexical` exercises the
complete pipeline without a model server. `provider=ollama` additionally requires a reachable
Ollama endpoint with the documented models pulled. Evaluation in this pipeline is explicitly
lexical; use the evaluation extra and the runbook for RAGAS judging of recorded outputs.

Compilation is tested independently of cluster execution. Confirm artifact staging, image
pull access, permissions, and network connectivity in your own OpenShift AI installation.
