import json
import sys

import pytest

from mobility_ai.capstone import app
from mobility_ai.evals import ragas_harness
from mobility_ai.phase9 import capstone as comparison


def invoke(monkeypatch, main, *args):
    monkeypatch.setattr(sys, "argv", ["command", *map(str, args)])
    main()


def test_documented_workflow_and_recorded_comparison(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "documents"
    corpus.mkdir()
    (corpus / "facts.md").write_text("pgvector adds vector search to PostgreSQL.")
    db = tmp_path / "corpus.db"
    invoke(monkeypatch, app.main, "ingest", "--corpus", corpus, "--db", db, "--provider", "lexical")
    assert json.loads(capsys.readouterr().out)["chunks"] == 1
    invoke(monkeypatch, app.main, "ask", "--db", db, "--question", "pgvector?")
    assert json.loads(capsys.readouterr().out)["citations"] == {"1": "facts.md#chunk-0"}
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question": "pgvector?", "ground_truth": "Vector search."}))
    records, report = tmp_path / "records.jsonl", tmp_path / "report.json"
    invoke(
        monkeypatch,
        app.main,
        "evaluate",
        "--db",
        db,
        "--eval-set",
        questions,
        "--records",
        records,
        "--output",
        report,
        "--backend",
        "lexical",
        "--revision",
        "test",
    )
    first = json.loads(report.read_text())
    assert first["configuration"]["generator"] == "extractive-v1"
    assert first["configuration"]["corpus_sha256"]
    assert first["per_sample"][0]["sample"]["cost_usd"] is None
    assert first["per_sample"][0]["sample"]["abstained"] is False
    assert first["per_sample"][0]["sample"]["citations"] == {"1": "facts.md#chunk-0"}
    invoke(
        monkeypatch,
        ragas_harness.main,
        "--eval-set",
        records,
        "--output",
        report,
        "--backend",
        "lexical",
    )
    assert json.loads(report.read_text())["metrics"] == first["metrics"]
    invoke(
        monkeypatch,
        comparison.main,
        "--run",
        f"first={records}",
        "--run",
        f"repeat={records}",
        "--output",
        report,
        "--backend",
        "lexical",
    )
    results = json.loads(report.read_text())["results"]
    assert results[0]["evaluation"]["metrics"] == results[1]["evaluation"]["metrics"]
    assert results[0]["mean_latency_ms"] >= 0


def test_cli_expansion_settings_reach_answers_and_evaluation(tmp_path, monkeypatch, capsys):
    doc = tmp_path / "facts.txt"
    text = "preface " * 62 + "namespace identifies model, prompt, corpus version, and tenant."
    doc.write_text(text)
    db = tmp_path / "db"
    invoke(monkeypatch, app.main, "ingest", "--corpus", doc, "--db", db, "--provider", "lexical")
    capsys.readouterr()
    settings = ["--top-k", "1", "--adjacent-chunks", "1", "--max-context-chars", "1000"]
    invoke(monkeypatch, app.main, "ask", "--db", db, "--question", "namespace", *settings)
    answer = json.loads(capsys.readouterr().out)
    assert answer["citations"] == {"1": "facts.txt#chunks-0-1"}
    assert answer["retrieved"][0]["text"] == text
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question": "namespace", "ground_truth": "model, prompt"}))
    records, report = tmp_path / "records.jsonl", tmp_path / "report.json"
    invoke(
        monkeypatch,
        app.main,
        "evaluate",
        "--db",
        db,
        "--eval-set",
        questions,
        "--records",
        records,
        "--output",
        report,
        "--backend",
        "lexical",
        *settings,
    )
    config = json.loads(report.read_text())["configuration"]
    assert (config["top_k"], config["adjacent_chunks"], config["max_context_chars"]) == (
        "1",
        "1",
        "1000",
    )
    recorded = json.loads(records.read_text())
    assert recorded["contexts"] == [text]
    assert recorded["citations"] == answer["citations"]


@pytest.mark.parametrize("runs", [["bad"], ["a=FILE", "a=FILE"]])
def test_comparison_cli_rejects_bad_run_names(tmp_path, monkeypatch, runs):
    records = tmp_path / "records.jsonl"
    records.write_text(
        ragas_harness.EvalSample(
            question="q", ground_truth="a", contexts=[], answer="a"
        ).model_dump_json()
    )
    args = [part for item in runs for part in ("--run", item.replace("FILE", str(records)))]
    with pytest.raises(SystemExit) as error:
        invoke(
            monkeypatch,
            comparison.main,
            *args,
            "--backend",
            "lexical",
            "--output",
            tmp_path / "out",
        )
    assert error.value.code == 2
