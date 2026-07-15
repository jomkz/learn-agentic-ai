"""Tests for mlflow_tracking.py with mocked mlflow module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def _make_mock_mlflow(run_id: str = "run-abc-123") -> MagicMock:
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = run_id
    mock.start_run.return_value = mock_run

    mock_version = MagicMock()
    mock_version.version = "3"
    mock.register_model.return_value = mock_version

    return mock


def test_log_finetuning_run_with_mock_mlflow() -> None:
    mock_mlflow = _make_mock_mlflow("run-xyz")

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        import importlib

        import mlflow_tracking

        importlib.reload(mlflow_tracking)

        config = mlflow_tracking.ExperimentConfig(
            experiment_name="test-exp",
            run_name="test-run",
            params={"lr": 0.001},
            tags={"env": "test"},
        )
        run_id = mlflow_tracking.log_finetuning_run(config, {"loss": 0.5})

    assert run_id == "run-xyz"
    mock_mlflow.set_experiment.assert_called_once_with("test-exp")
    mock_mlflow.log_params.assert_called_once_with({"lr": 0.001})
    mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5})
    mock_mlflow.set_tag.assert_called_with("env", "test")


def test_log_finetuning_run_with_tags() -> None:
    mock_mlflow = _make_mock_mlflow()

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        import importlib

        import mlflow_tracking

        importlib.reload(mlflow_tracking)

        config = mlflow_tracking.ExperimentConfig(
            experiment_name="e",
            run_name="r",
            params={},
            tags={"k1": "v1", "k2": "v2"},
        )
        mlflow_tracking.log_finetuning_run(config, {})

    assert mock_mlflow.set_tag.call_count == 2


def test_register_model_with_mock_mlflow() -> None:
    mock_mlflow = _make_mock_mlflow()

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        import importlib

        import mlflow_tracking

        importlib.reload(mlflow_tracking)

        version = mlflow_tracking.register_model("run-abc", "my-model")

    assert version == "3"
    mock_mlflow.register_model.assert_called_once_with("runs:/run-abc/model", "my-model")
