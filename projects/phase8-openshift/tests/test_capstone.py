from __future__ import annotations

import asyncio
from unittest.mock import patch

from capstone import DVC_PIPELINE_YAML, ExperimentTracker, ModelVersion, VLLMClient


def test_model_version_default_status():
    mv = ModelVersion(run_id="r", version="1", model_name="m", metrics={}, params={})
    assert mv.status == "staging"


def test_model_version_promote():
    mv = ModelVersion(run_id="r", version="1", model_name="m", metrics={}, params={})
    mv.promote()
    assert mv.status == "production"


def test_model_version_archive():
    mv = ModelVersion(run_id="r", version="1", model_name="m", metrics={}, params={})
    mv.archive()
    assert mv.status == "archived"


def test_tracker_empty_best():
    assert ExperimentTracker().best_run() is None


def test_tracker_track_run():
    with patch("capstone.log_finetuning_run", return_value="run-abc"), \
         patch("capstone.register_model", return_value="1"):
        t = ExperimentTracker()
        mv = t.track_run("r1", {"lr": 0.001}, {"eval_loss": 0.4})
        assert isinstance(mv, ModelVersion)


def test_tracker_best_run():
    with patch("capstone.log_finetuning_run", side_effect=["run-1", "run-2"]), \
         patch("capstone.register_model", side_effect=["1", "2"]):
        t = ExperimentTracker()
        t.track_run("r1", {}, {"eval_loss": 0.4})
        t.track_run("r2", {}, {"eval_loss": 0.3})
        assert t.best_run().metrics["eval_loss"] == 0.3


def test_quality_gate_passes():
    mv = ModelVersion(
        run_id="r", version="1", model_name="m", metrics={"eval_loss": 0.3}, params={}
    )
    assert ExperimentTracker().quality_gate(mv) is True


def test_quality_gate_fails():
    mv = ModelVersion(
        run_id="r", version="1", model_name="m", metrics={"eval_loss": 0.8}, params={}
    )
    assert ExperimentTracker().quality_gate(mv) is False


def test_vllm_client_defaults():
    assert VLLMClient().base_url == "http://localhost:8000/v1"


def test_vllm_client_complete_offline():
    result = asyncio.run(VLLMClient().complete("test"))
    assert isinstance(result, str)


def test_dvc_pipeline_is_str():
    assert isinstance(DVC_PIPELINE_YAML, str)


def test_dvc_pipeline_has_stages():
    assert "stages" in DVC_PIPELINE_YAML and "train" in DVC_PIPELINE_YAML
