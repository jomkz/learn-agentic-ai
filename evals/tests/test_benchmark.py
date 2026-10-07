import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from mobility_ai.capstone.app import ABSTENTION
from mobility_ai.evals import benchmark


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "suite"
    (root / "corpus").mkdir(parents=True)
    doc = root / "corpus" / "facts.txt"
    doc.write_text("Vectors support similarity search.")
    benchmark.write_json(
        root / "corpus-manifest.json", {"facts.txt": {"sha256": benchmark.digest(doc)}}
    )
    questions = [
        dict(
            id="q1",
            category="retrieval",
            question="What do vectors support?",
            ground_truth="Similarity search.",
            answerable=True,
            evidence=[dict(source="facts.txt", text="similarity search")],
        ),
        dict(
            id="q2",
            category="unanswerable",
            question="What is the password?",
            ground_truth=ABSTENTION,
            answerable=False,
            evidence=[],
        ),
    ]
    (root / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    return root


@pytest.fixture
def ollama(monkeypatch):
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "generator", "digest": "g"},
                        {"name": "embedder", "digest": "e"},
                    ]
                },
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in payload["input"]]})
        assert payload["options"]["seed"] == 0
        abstain = "password" in payload["messages"][-1]["content"]
        text = json.dumps(
            {
                "answer": "Not enough information." if abstain else "Similarity search.",
                "citations": [] if abstain else [1],
                "abstain": abstain,
            }
        )
        return httpx.Response(200, json={"message": {"content": text}, "eval_count": 5})

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(**(kwargs | {"transport": httpx.MockTransport(handler)})),
    )


@pytest.mark.parametrize("adjacent_chunks", [0, 1])
def test_capture_cli_preserves_every_case_and_provenance(
    suite, ollama, tmp_path, monkeypatch, adjacent_chunks
):
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "capture",
            "--suite",
            str(suite),
            "--output",
            str(output),
            "--generation-model",
            "generator",
            "--embedding-model",
            "embedder",
            "--adjacent-chunks",
            str(adjacent_chunks),
            "--max-context-chars",
            "1000",
        ],
    )
    benchmark.main()
    summary = json.loads((output / "summary.json").read_text())
    for top_k in (1, 3):
        assert summary[f"top{top_k}"]["total"] == 2
        assert summary[f"top{top_k}"]["unanswerable_exact_abstentions"] == 1
        assert summary[f"top{top_k}"]["answerable_with_all_reference_evidence"] == 1
        rows = benchmark.read_jsonl(output / f"top{top_k}.jsonl")
        assert rows[0]["result"]["citations"] == {"1": "facts.txt#chunk-0"}
        assert rows[0]["adjacent_chunks"] == adjacent_chunks
        assert rows[0]["result"]["adjacent_chunks"] == adjacent_chunks
        assert rows[0]["max_context_chars"] == 1000
        assert rows[0]["provider_calls"][-1]["statistics"]["eval_count"] == 5
        assert json.loads(rows[0]["raw_response"])["citations"] == [1]
        assert rows[0]["abstained"] is False
        assert rows[1]["abstained"] is True
        assert benchmark.sample(rows[0]).citations == rows[0]["result"]["citations"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["models"]["generator"]["digest"] == "g"
    assert metadata["completed_at"]
    assert metadata["source_sha256"]
    assert metadata["adjacent_chunks"] == adjacent_chunks
    assert metadata["max_context_chars"] == 1000
    with pytest.raises(FileExistsError):
        benchmark.capture(suite, output, "http://localhost", "generator", "embedder")


@pytest.mark.parametrize(
    "raw",
    [
        "  Unsupported uncited answer\n",
        json.dumps({"answer": "Answer.", "citations": [], "abstain": False}),
        '{"answer":',
    ],
)
def test_rejected_answer_is_preserved(suite, ollama, tmp_path, monkeypatch, raw):
    client = benchmark.RecordingClient()
    client.reset()
    db = tmp_path / "db"
    benchmark.ingest(
        suite / "corpus",
        db,
        embedding_model="embedder",
        generation_model="generator",
        client=client,
    )
    original = client.request

    def uncited(path, payload):
        response = original(path, payload)
        if path == "/api/chat":
            response["message"]["content"] = raw
        return response

    monkeypatch.setattr(client, "request", uncited)
    result = benchmark.capture_one(db, benchmark.load_suite(suite)[0], 1, client)
    assert result["status"] == "error"
    assert result["answer"] == ""
    assert result["raw_response"] == raw
    assert result["abstained"] is None
    assert result["citations"] is None
    assert result["contexts"] == ["Vectors support similarity search."]
    assert benchmark.summarize([result])["application_errors"] == 1


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"answer": ABSTENTION}, True),
        ({"answer": "Answer [1]"}, False),
        ({"answer": "Different wording", "abstained": True}, True),
        ({"answer": ABSTENTION, "abstained": None}, False),
        ({"answer": ABSTENTION, "abstained": False}, False),
    ],
)
def test_abstention_state_supports_frozen_legacy_records(row, expected):
    assert benchmark.is_abstention(row) is expected


@pytest.mark.parametrize("change", ["corpus", "duplicate", "answerable", "evidence"])
def test_suite_rejects_invalid_or_changed_inputs(suite, change):
    rows = benchmark.load_suite(suite)
    if change == "corpus":
        (suite / "corpus" / "facts.txt").write_text("Changed")
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "answerable":
        rows[0]["answerable"] = False
    else:
        rows[0]["evidence"][0]["text"] = "not in the document"
    (suite / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in rows))
    with pytest.raises(ValueError):
        benchmark.load_suite(suite)


def test_judgments_report_failures_and_exclusions(tmp_path, monkeypatch, ollama):
    rows = [
        dict(
            id=f"q{i}",
            top_k=1,
            question=f"question {i}",
            ground_truth="reference",
            contexts=["context"],
            answer="answer" if i < 2 else ABSTENTION,
            answerable=i < 2,
            latency_ms=10,
        )
        for i in range(3)
    ]
    for k in (1, 3):
        (tmp_path / f"top{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    benchmark.write_json(tmp_path / "metadata.json", {"completed_at": "test"})
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake)
    monkeypatch.setitem(sys.modules, "ragas.run_config", SimpleNamespace(RunConfig=MagicMock()))
    report = SimpleNamespace(model_dump=lambda: {"metrics": {"faithfulness": 0.5}})
    call = MagicMock(side_effect=[report, ValueError("bad judge output"), report, report])
    monkeypatch.setattr(benchmark, "compute_report", call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "judge",
            "--output",
            str(tmp_path),
            "--judge-model",
            "generator",
            "--embedding-model",
            "embedder",
        ],
    )
    benchmark.main()
    summary = json.loads((tmp_path / "judge-summary.json").read_text())
    assert summary["top1"] == {
        "scored": 1,
        "errors": 1,
        "not_applicable": 1,
        "conditional_mean_metrics": {"faithfulness": 0.5},
    }
    assert len(benchmark.read_jsonl(tmp_path / "judgments.jsonl")) == 6
    with pytest.raises(ValueError, match="already exist"):
        benchmark.main()


def test_frozen_real_suite_has_24_grounded_or_unanswerable_cases():
    questions = benchmark.load_suite(Path("evals/benchmarks/project-docs-v1"))
    assert len(questions) == 24
    assert sum(q["answerable"] for q in questions) == 20
