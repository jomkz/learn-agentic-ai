"""Exercise the installed evaluation SDK; mocks must not hide import incompatibilities."""

import importlib.util

import pytest


def test_real_evaluation_sdk_imports(monkeypatch):
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("Install the evaluation extra for the real SDK import check")
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    assert callable(evaluate)
    assert [m.name for m in (faithfulness, answer_relevancy, context_precision)] == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]
