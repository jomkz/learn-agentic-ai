from __future__ import annotations

import pytest
from monitoring import (
    RETRAINING_TRIGGER,
    DriftResult,
    MonitoringConfig,
    compute_text_drift,
    run_evidently_report,
)


def test_monitoring_config_defaults() -> None:
    config = MonitoringConfig(reference_path="r", production_path="p")
    assert config.drift_threshold == 0.1


def test_drift_result_model() -> None:
    result = DriftResult(has_drift=False, drift_score=0.05, recommendation="ok")
    assert result.has_drift is False
    assert result.drift_score == 0.05
    assert result.recommendation == "ok"


def test_no_drift_identical_texts() -> None:
    result = compute_text_drift(["hello world"] * 5, ["hello world"] * 5)
    assert result.has_drift is False


def test_drift_score_zero_for_identical() -> None:
    result = compute_text_drift(["hello"] * 5, ["hello"] * 5)
    assert result.drift_score == pytest.approx(0.0, abs=0.01)


def test_no_drift_similar_texts() -> None:
    reference = ["The model processes requests and returns predictions."] * 5
    production = ["The model handles requests and returns prediction results."] * 5
    result = compute_text_drift(reference, production)
    assert result.has_drift is False


def test_drift_empty_reference() -> None:
    result = compute_text_drift([], ["some text"])
    assert isinstance(result, DriftResult)


def test_drift_empty_production() -> None:
    result = compute_text_drift(["reference text"], [])
    assert isinstance(result, DriftResult)


def test_retraining_trigger_has_threshold() -> None:
    assert "drift_threshold" in RETRAINING_TRIGGER


def test_retraining_trigger_has_pipeline_ref() -> None:
    assert "pipeline_ref" in RETRAINING_TRIGGER


def test_run_evidently_report_returns_string() -> None:
    config = MonitoringConfig(reference_path="r", production_path="p")
    result = run_evidently_report(config)
    assert isinstance(result, str)


def test_drift_recommendation_has_content() -> None:
    result = compute_text_drift(["a"] * 5, ["a"] * 5)
    assert result.recommendation != ""
