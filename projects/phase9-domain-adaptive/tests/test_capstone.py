import pytest

from mobility_ai.evals.ragas_harness import EvalSample
from mobility_ai.phase9.capstone import run_comparison


def sample(answer="Paris", latency=None):
    return EvalSample(
        question="Capital of France?",
        ground_truth="Paris",
        contexts=["Paris"],
        answer=answer,
        latency_ms=latency,
    )


def test_no_assumed_advantage_for_named_strategies():
    report = run_comparison({"Naive RAG": [sample()], "RAFT": [sample()]}, backend="lexical")
    assert report.results[0].evaluation.metrics == report.results[1].evaluation.metrics
    assert report.results[0].mean_latency_ms is None
    assert report.results[0].mean_cost_usd is None
    assert "not recorded" in report.to_markdown_table()


def test_measured_outputs_determine_metrics():
    report = run_comparison(
        {"baseline": [sample()], "candidate": [sample("London", 42)]}, backend="lexical"
    )
    assert report.results[0].evaluation.metrics["answer_exact_match"] == 1
    assert report.results[1].evaluation.metrics["answer_exact_match"] == 0
    assert report.results[1].mean_latency_ms == 42


def test_mismatched_questions_rejected():
    other = sample().model_copy(update={"question": "Different question"})
    with pytest.raises(ValueError, match="same questions"):
        run_comparison({"a": [sample()], "b": [other]}, backend="lexical")


def test_missing_strategy_rejected():
    with pytest.raises(ValueError, match="at least two"):
        run_comparison({"a": [sample()]}, backend="lexical")
