"""Production drift monitoring with Evidently AI. Run batch job to detect quality degradation."""

from __future__ import annotations

from pydantic import BaseModel


class MonitoringConfig(BaseModel):
    reference_path: str
    production_path: str
    output_path: str = "./drift_report.html"
    drift_threshold: float = 0.1


class DriftResult(BaseModel):
    has_drift: bool
    drift_score: float
    recommendation: str


def compute_text_drift(reference_texts: list[str], production_texts: list[str]) -> DriftResult:
    def avg_len(texts: list[str]) -> float:
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / len(texts)

    def vocab(texts: list[str]) -> set[str]:
        words: set[str] = set()
        for t in texts:
            words.update(t.lower().split())
        return words

    ref_len = avg_len(reference_texts)
    prod_len = avg_len(production_texts)
    avg_len_ratio = prod_len / ref_len if ref_len > 0 else 1.0

    ref_vocab = vocab(reference_texts)
    prod_vocab = vocab(production_texts)
    if ref_vocab | prod_vocab:
        vocab_overlap = len(ref_vocab & prod_vocab) / len(ref_vocab | prod_vocab)
    else:
        vocab_overlap = 1.0

    has_drift = abs(avg_len_ratio - 1) > 0.3 or vocab_overlap < 0.5
    drift_score = max(abs(avg_len_ratio - 1), 1 - vocab_overlap)

    if has_drift:
        recommendation = (
            "Drift detected. Review production data distribution and consider retraining."
        )
    else:
        recommendation = "No significant drift detected. Continue monitoring."

    return DriftResult(has_drift=has_drift, drift_score=drift_score, recommendation=recommendation)


def run_evidently_report(config: MonitoringConfig) -> str:
    try:
        import evidently  # noqa: F401
        from evidently.metric_preset import DataDriftPreset  # noqa: F401
        from evidently.report import Report  # noqa: F401

        report = Report(metrics=[DataDriftPreset()])
        report.save_html(config.output_path)
        return config.output_path
    except ImportError:
        return "[evidently not installed — install with: pip install evidently]"


RETRAINING_TRIGGER: dict = {
    "drift_threshold": 0.1,
    "evaluation_window_days": 7,
    "min_production_samples": 100,
    "pipeline_ref": "openshift-ai-pipeline://retrain-domain-model",
    "alert_channel": "slack://mlops-alerts",
    "metrics": ["ragas_faithfulness", "ragas_answer_relevancy", "token_length_drift"],
}


if __name__ == "__main__":
    import json

    reference = [
        "OpenShift AI manages ML workloads on Kubernetes.",
        "vLLM serves large language models efficiently.",
        "KubeFlow Pipelines automates ML training jobs.",
    ] * 5

    production_similar = [
        "OpenShift AI manages ML workflows on Kubernetes clusters.",
        "vLLM efficiently serves large models with paged attention.",
        "KubeFlow automates machine learning pipeline jobs.",
    ] * 5

    production_drifted = [
        "Very very very long detailed technical document with many many words and verbose concepts"
        " and explanations that go on and on and contain much more information than before."
    ] * 10

    result_no_drift = compute_text_drift(reference, production_similar)
    result_drift = compute_text_drift(reference, production_drifted)

    print("No-drift scenario:")
    print(json.dumps(result_no_drift.model_dump(), indent=2))
    print("\nDrift scenario:")
    print(json.dumps(result_drift.model_dump(), indent=2))
    print("\nRetraining trigger config:")
    print(json.dumps(RETRAINING_TRIGGER, indent=2))
