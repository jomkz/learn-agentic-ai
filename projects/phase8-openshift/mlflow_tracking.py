"""MLflow experiment tracking for fine-tuning runs. Requires mlflow."""

from __future__ import annotations

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    experiment_name: str
    run_name: str
    params: dict
    tags: dict = {}


def log_finetuning_run(
    config: ExperimentConfig,
    metrics: dict[str, float],
    artifact_path: str | None = None,
) -> str:
    try:
        import mlflow

        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.log_params(config.params)
            mlflow.log_metrics(metrics)
            [mlflow.set_tag(k, v) for k, v in config.tags.items()]
            return run.info.run_id
    except ImportError:
        return "mlflow-not-installed"


def register_model(run_id: str, model_name: str, artifact_path: str = "model") -> str:
    try:
        import mlflow

        result = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
        return result.version
    except Exception:
        return "0"


if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="qlora-llama3",
        run_name="baseline-run",
        params={"lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4, "num_epochs": 3},
        tags={"model": "llama-3.2-3b", "task": "instruction-tuning"},
    )
    metrics = {"train_loss": 0.42, "eval_loss": 0.51, "perplexity": 6.3}
    run_id = log_finetuning_run(config, metrics)
    print(f"Run ID: {run_id}")
