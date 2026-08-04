from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "evals"))

from ragas_harness import EvalSample  # noqa: E402

from capstone import (  # noqa: E402
    ApproachResult,
    ComparisonReport,
    EVAL_QUESTIONS,
    simulate_naive_rag,
    simulate_graphrag,
    simulate_raft,
    run_comparison,
)


def test_approach_result_model():
    result = ApproachResult(
        approach="RAG",
        ragas_faithfulness=0.7,
        ragas_relevancy=0.8,
        ragas_precision=0.75,
        latency_ms=150.0,
        cost_per_query_usd=0.001,
    )
    assert result.approach == "RAG"
    assert result.ragas_faithfulness == 0.7


def test_eval_questions_count():
    assert len(EVAL_QUESTIONS) == 5


def test_eval_questions_are_eval_samples():
    assert all(isinstance(q, EvalSample) for q in EVAL_QUESTIONS)


def test_comparison_report_best_approach():
    r = ComparisonReport(
        domain="d",
        question_count=2,
        results=[
            ApproachResult(
                approach="A",
                ragas_faithfulness=0.8,
                ragas_relevancy=0.7,
                ragas_precision=0.75,
                latency_ms=100,
                cost_per_query_usd=0.001,
            ),
            ApproachResult(
                approach="B",
                ragas_faithfulness=0.9,
                ragas_relevancy=0.8,
                ragas_precision=0.8,
                latency_ms=200,
                cost_per_query_usd=0.002,
            ),
        ],
        recommendation="use B",
        generated_at="2024",
    )
    assert r.best_approach().approach == "B"


def test_comparison_report_markdown_has_approach_col():
    report = ComparisonReport(
        domain="d",
        question_count=1,
        results=[],
        recommendation="x",
        generated_at="y",
    )
    assert "Approach" in report.to_markdown_table()


def test_simulate_naive_rag():
    assert isinstance(simulate_naive_rag(EVAL_QUESTIONS), ApproachResult)


def test_simulate_graphrag_higher_than_naive():
    assert (
        simulate_graphrag(EVAL_QUESTIONS).ragas_faithfulness
        >= simulate_naive_rag(EVAL_QUESTIONS).ragas_faithfulness
    )


def test_simulate_raft_highest_faithfulness():
    assert (
        simulate_raft(EVAL_QUESTIONS).ragas_faithfulness
        >= simulate_naive_rag(EVAL_QUESTIONS).ragas_faithfulness
    )


def test_run_comparison_is_report():
    assert isinstance(run_comparison(), ComparisonReport)


def test_run_comparison_four_results():
    assert len(run_comparison().results) == 4


def test_run_comparison_recommendation_non_empty():
    assert len(run_comparison().recommendation) > 10
