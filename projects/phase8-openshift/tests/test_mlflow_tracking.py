from __future__ import annotations

from mlflow_tracking import ExperimentConfig, log_finetuning_run, register_model


def test_log_run_without_mlflow():
    config = ExperimentConfig(
        experiment_name="test-experiment",
        run_name="test-run",
        params={"lr": 0.001, "epochs": 3},
    )
    metrics = {"train_loss": 0.5, "eval_loss": 0.6}
    result = log_finetuning_run(config, metrics)
    assert isinstance(result, str)
    assert len(result) > 0


def test_experiment_config_validates():
    config = ExperimentConfig(
        experiment_name="test",
        run_name="run1",
        params={"lr": 0.001},
    )
    assert config.experiment_name == "test"
    assert config.run_name == "run1"
    assert config.params == {"lr": 0.001}
    assert config.tags == {}


def test_register_model_without_mlflow():
    result = register_model("fake-run-id", "test-model")
    assert isinstance(result, str)
