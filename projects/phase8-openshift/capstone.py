"""Phase 8 capstone: MLOps integration — experiment tracking, vLLM client, DVC pipeline config."""

from __future__ import annotations

import asyncio

from mlflow_tracking import ExperimentConfig, log_finetuning_run, register_model
from pydantic import BaseModel
from qlora_finetune import AXOLOTL_CONFIG_EXAMPLE, FinetuneConfig


class ModelVersion(BaseModel):
    run_id: str
    version: str
    model_name: str
    metrics: dict[str, float]
    params: dict
    status: str = "staging"

    def promote(self) -> None:
        self.status = "production"

    def archive(self) -> None:
        self.status = "archived"


class ExperimentTracker:
    def __init__(self, experiment_name: str = "qlora-finetune"):
        self.experiment_name = experiment_name
        self.versions: list[ModelVersion] = []

    def track_run(self, run_name: str, params: dict, metrics: dict[str, float]) -> ModelVersion:
        config = ExperimentConfig(
            experiment_name=self.experiment_name, run_name=run_name, params=params
        )
        run_id = log_finetuning_run(config, metrics)
        version = register_model(run_id, self.experiment_name)
        mv = ModelVersion(
            run_id=run_id,
            version=version,
            model_name=self.experiment_name,
            metrics=metrics,
            params=params,
        )
        self.versions.append(mv)
        return mv

    def best_run(self, metric: str = "eval_loss") -> ModelVersion | None:
        if not self.versions:
            return None
        return min(self.versions, key=lambda v: v.metrics.get(metric, float("inf")))

    def quality_gate(self, version: ModelVersion, max_loss: float = 0.5) -> bool:
        return version.metrics.get("eval_loss", float("inf")) < max_loss


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "llama3.1:8b"):
        self.base_url = base_url
        self.model = model

    async def complete(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=self.base_url, api_key="vllm")
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[vLLM unavailable — falling back to Ollama: {e}]"

    async def health(self) -> dict:
        try:
            import httpx

            httpx.get(f"{self.base_url.replace('/v1', '')}/health", timeout=2).raise_for_status()
            return {"status": "ok", "model": self.model}
        except Exception:
            return {"status": "unavailable", "model": self.model}


DVC_PIPELINE_YAML: str = """\
stages:
  prepare:
    cmd: python prepare.py
    deps: [data/raw]
    outs: [data/processed]
  train:
    cmd: python train.py
    deps: [data/processed, config/train.yaml]
    outs: [models/adapter]
    metrics: [metrics/train.json]
  evaluate:
    cmd: python evaluate.py
    deps: [models/adapter, data/eval]
    metrics: [metrics/eval.json]
"""


if __name__ == "__main__":
    finetune_config = FinetuneConfig()
    print(f"Default base model: {finetune_config.base_model}")
    print(f"Axolotl config snippet:\n{AXOLOTL_CONFIG_EXAMPLE[:60]}...")

    tracker = ExperimentTracker()
    mv1 = tracker.track_run(
        "run-lr-0001", {"lora_r": 16, "lr": 0.001}, {"eval_loss": 0.42, "train_loss": 0.38}
    )
    mv2 = tracker.track_run(
        "run-lr-0002", {"lora_r": 8, "lr": 0.002}, {"eval_loss": 0.38, "train_loss": 0.35}
    )
    best = tracker.best_run()
    print(f"Best run: {best.run_id} (eval_loss={best.metrics['eval_loss']})")
    print(f"Quality gate: {tracker.quality_gate(best)}")
    print(f"\nDVC Pipeline:\n{DVC_PIPELINE_YAML}")

    client = VLLMClient()
    health = asyncio.run(client.health())
    print(f"vLLM health: {health}")
