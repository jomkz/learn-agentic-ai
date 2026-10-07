from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from mobility_ai.evals.ragas_harness import EvalSample, compute_report, load_samples, save_report


def sample(answer="Paris is the capital of France."):
    return EvalSample(
        question="What is the capital of France?",
        ground_truth="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
        answer=answer,
    )


def test_lexical_metrics_do_not_claim_faithfulness():
    report = compute_report([sample("London is the capital of France.")], backend="lexical")
    assert "faithfulness" not in report.metrics
    assert report.metrics["answer_exact_match"] == 0
    assert report.metrics["answer_token_f1"] < 1
    assert report.backend == "lexical"


def test_exact_answer_and_provenance(tmp_path):
    report = compute_report([sample()], backend="lexical", configuration={"revision": "abc"})
    assert report.metrics["answer_exact_match"] == 1
    assert report.per_sample[0].sample.answer == sample().answer
    assert report.configuration["revision"] == "abc"
    path = tmp_path / "nested" / "report.json"
    save_report(report, path)
    assert json.loads(path.read_text())["dataset_sha256"] == report.dataset_sha256


def test_dataset_hash_changes_with_outputs():
    a = compute_report([sample()], backend="lexical")
    b = compute_report([sample("unknown")], backend="lexical")
    assert a.dataset_sha256 != b.dataset_sha256


def test_default_requires_explicit_judge():
    with pytest.raises(ValueError, match="explicit"):
        compute_report([sample()])


def test_missing_ragas_never_falls_back():
    with patch.dict(sys.modules, {"ragas": None, "datasets": None}):
        with pytest.raises(RuntimeError, match="evaluation extra"):
            compute_report(
                [sample()],
                llm=object(),
                embeddings=object(),
                judge_id="judge",
                embedding_id="embed",
            )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_judge_scores_rejected(invalid):
    datasets, ragas, metrics = MagicMock(), MagicMock(), MagicMock()
    ragas.evaluate.return_value.scores = [
        {"faithfulness": invalid, "answer_relevancy": 0.5, "context_precision": 0.5}
    ]
    with patch.dict(sys.modules, {"datasets": datasets, "ragas": ragas, "ragas.metrics": metrics}):
        with pytest.raises(ValueError, match="invalid scores"):
            compute_report(
                [sample()],
                llm=object(),
                embeddings=object(),
                judge_id="judge",
                embedding_id="embed",
            )
    assert ragas.evaluate.call_args.kwargs["raise_exceptions"] is True


def test_empty_samples_rejected():
    with pytest.raises(ValueError, match="at least one"):
        compute_report([], backend="lexical")


def test_duplicate_questions_rejected(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text((sample().model_dump_json() + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        load_samples(path)


def test_successful_ragas_uses_explicit_clients_and_records_scores():
    datasets, ragas, metrics = MagicMock(), MagicMock(), MagicMock()
    ragas.evaluate.return_value.scores = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6}
    ]
    judge, embedder = object(), object()
    recorded = sample().model_copy(update={"abstained": False, "citations": {"1": "facts#chunk-0"}})
    with patch.dict(sys.modules, {"datasets": datasets, "ragas": ragas, "ragas.metrics": metrics}):
        result = compute_report(
            [recorded],
            llm=judge,
            embeddings=embedder,
            judge_id="local-judge",
            embedding_id="local-embedder",
        )
    assert result.backend == "ragas"
    assert result.metrics["faithfulness"] == 0.8
    assert result.configuration["judge"] == "local-judge"
    assert ragas.evaluate.call_args.kwargs["llm"] is judge
    assert ragas.evaluate.call_args.kwargs["embeddings"] is embedder
    assert result.per_sample[0].sample.citations == {"1": "facts#chunk-0"}
    datasets.Dataset.from_list.assert_called_once_with(
        [
            {
                "question": recorded.question,
                "ground_truth": recorded.ground_truth,
                "contexts": recorded.contexts,
                "answer": recorded.answer,
            }
        ]
    )


def test_ragas_cli_options_configure_local_models():
    from argparse import Namespace

    from mobility_ai.evals.ragas_harness import evaluation_options

    module = MagicMock()
    with patch.dict(sys.modules, {"langchain_ollama": module}):
        options = evaluation_options(
            Namespace(
                backend="ragas",
                revision="rev",
                judge_model="judge",
                embedding_model="embed",
                ollama_url="http://localhost:11434",
            )
        )
    assert options["judge_id"] == "judge"
    module.ChatOllama.assert_called_once_with(
        model="judge", base_url="http://localhost:11434", temperature=0
    )
    module.OllamaEmbeddings.assert_called_once_with(
        model="embed", base_url="http://localhost:11434"
    )


def test_invalid_backend_and_empty_file(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        compute_report([sample()], backend="other")
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_samples(path)


@pytest.mark.parametrize("value, expected", [(1.0000000000000002, 1.0), (-0.2, -0.2)])
def test_cosine_metric_preserves_range_and_tolerates_roundoff(value, expected):
    datasets, ragas, metrics = MagicMock(), MagicMock(), MagicMock()
    ragas.evaluate.return_value.scores = [
        {"faithfulness": 1.0, "answer_relevancy": value, "context_precision": 1.0}
    ]
    with patch.dict(sys.modules, {"datasets": datasets, "ragas": ragas, "ragas.metrics": metrics}):
        report = compute_report(
            [sample()], llm=object(), embeddings=object(), judge_id="judge", embedding_id="embed"
        )
    assert report.metrics["answer_relevancy"] == expected
    assert report.configuration["evaluator_version"] == "3"


def test_legacy_records_load_without_structured_metadata(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(sample().model_dump_json(exclude={"abstained", "citations"}) + "\n")
    loaded = load_samples(path)[0]
    assert loaded.abstained is None
    assert loaded.citations is None
