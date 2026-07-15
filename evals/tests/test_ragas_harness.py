"""Tests for evals/ragas_harness.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ragas_harness import EvalSample, RAGASReport, _heuristic_report, compute_report, save_report


def _sample(
    question: str = "What is RAG?",
    ground_truth: str = "RAG is Retrieval-Augmented Generation.",
    contexts: list[str] | None = None,
    answer: str = "RAG stands for Retrieval-Augmented Generation.",
) -> EvalSample:
    if contexts is None:
        contexts = ["Retrieval-Augmented Generation combines retrieval with generation."]
    return EvalSample(
        question=question, ground_truth=ground_truth, contexts=contexts, answer=answer
    )


def test_eval_sample_validates() -> None:
    s = _sample()
    assert s.question == "What is RAG?"
    assert s.ground_truth == "RAG is Retrieval-Augmented Generation."
    assert isinstance(s.contexts, list)
    assert s.answer == "RAG stands for Retrieval-Augmented Generation."


def test_ragas_report_fields() -> None:
    r = RAGASReport(
        faithfulness=0.8,
        answer_relevancy=0.7,
        context_precision=0.9,
        overall=0.8,
        sample_count=3,
    )
    assert hasattr(r, "faithfulness")
    assert hasattr(r, "answer_relevancy")
    assert hasattr(r, "context_precision")
    assert hasattr(r, "overall")
    assert hasattr(r, "sample_count")


def test_compute_report_returns_report() -> None:
    samples = [_sample()]
    result = compute_report(samples)
    assert isinstance(result, RAGASReport)


def test_heuristic_faithful_answer() -> None:
    # answer shares a word with ground_truth -> faithfulness > 0
    s = _sample(
        ground_truth="Paris is the capital of France.",
        answer="Paris is a city in France.",
    )
    report = _heuristic_report([s])
    assert report.faithfulness > 0.0


def test_heuristic_unfaithful_answer() -> None:
    # answer shares NO words with ground_truth -> faithfulness == 0.0
    # Use unique non-overlapping tokens
    s = EvalSample(
        question="xyz",
        ground_truth="alpha bravo charlie delta",
        contexts=["some context here"],
        answer="foxtrot golf hotel india",
    )
    report = _heuristic_report([s])
    assert report.faithfulness == 0.0


def test_heuristic_context_precision() -> None:
    # ground_truth words appear in contexts -> context_precision == 1.0
    s = _sample(
        ground_truth="vector similarity search",
        contexts=["pgvector enables vector similarity search in PostgreSQL"],
        answer="it does vector similarity search",
    )
    report = _heuristic_report([s])
    assert report.context_precision == 1.0


def test_heuristic_answer_relevancy() -> None:
    # answer shares words with question -> relevancy > 0
    s = _sample(
        question="What is chunking?",
        answer="Chunking is the process of splitting documents.",
    )
    report = _heuristic_report([s])
    assert report.answer_relevancy > 0.0


def test_compute_report_sample_count() -> None:
    samples = [_sample(), _sample(), _sample()]
    report = compute_report(samples)
    assert report.sample_count == 3


def test_save_report_creates_json(tmp_path: Path) -> None:
    report = RAGASReport(
        faithfulness=0.75,
        answer_relevancy=0.80,
        context_precision=0.90,
        overall=0.8167,
        sample_count=2,
    )
    out = tmp_path / "report.json"
    save_report(report, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert "faithfulness" in data
    assert "sample_count" in data


def test_save_report_roundtrip(tmp_path: Path) -> None:
    report = RAGASReport(
        faithfulness=0.6,
        answer_relevancy=0.7,
        context_precision=0.8,
        overall=0.7,
        sample_count=5,
    )
    out = tmp_path / "roundtrip.json"
    save_report(report, str(out))
    loaded = RAGASReport(**json.loads(out.read_text()))
    assert loaded.faithfulness == pytest.approx(0.6)
    assert loaded.answer_relevancy == pytest.approx(0.7)
    assert loaded.context_precision == pytest.approx(0.8)
    assert loaded.overall == pytest.approx(0.7)
    assert loaded.sample_count == 5


def test_empty_question_still_valid() -> None:
    s = EvalSample(
        question="",
        ground_truth="something",
        contexts=["context text"],
        answer="an answer",
    )
    report = _heuristic_report([s])
    assert isinstance(report, RAGASReport)
    # empty question -> relevancy falls to the q_words=0 branch -> 0.0
    assert report.answer_relevancy == 0.0


def test_overall_is_mean_of_three() -> None:
    samples = [_sample()]
    report = _heuristic_report(samples)
    expected = (report.faithfulness + report.answer_relevancy + report.context_precision) / 3.0
    assert report.overall == pytest.approx(expected)
