import json

import httpx
import pytest

from mobility_ai.capstone.app import (
    ABSTENTION,
    OllamaClient,
    VectorStore,
    ask,
    ingest,
    record_evaluation,
)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus"
    path.mkdir()
    (path / "database.txt").write_text("pgvector adds vector search to PostgreSQL.")
    (path / "agents.txt").write_text("Agents use tools to perform tasks.")
    return path


def test_persist_reopen_retrieve_and_cite(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    assert ingest(corpus, db, provider="lexical") == 2
    result = ask(db, "What does pgvector add?")
    assert result.citations == {"1": "database.txt#chunk-0"}
    assert "vector search" in result.answer
    assert result.latency_ms >= 0
    assert result.generation_model == "extractive-v1"
    assert result.abstained is False
    unknown = ask(db, "Jupiter moons")
    assert unknown.answer == ABSTENTION
    assert unknown.abstained is True
    assert unknown.citations == {}


def test_reingestion_replaces_changed_and_deleted_documents(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    ingest(corpus, db, provider="lexical")
    ingest(corpus, db, provider="lexical")
    assert len(VectorStore(db).load()[1]) == 2
    (corpus / "agents.txt").unlink()
    (corpus / "database.txt").write_text("Replaced content.")
    ingest(corpus, db, provider="lexical")
    assert len(VectorStore(db).load()[1]) == 1
    assert ask(db, "pgvector").answer == ABSTENTION


def test_ollama_requests_and_invalid_citations(corpus, tmp_path):
    def handle(request):
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            assert payload["truncate"] is False
            return httpx.Response(200, json={"embeddings": [[1, 0] for _ in payload["input"]]})
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {"answer": "Unsupported citation", "citations": [99], "abstain": False}
                    )
                }
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(handle))
    db = tmp_path / "corpus.db"
    ingest(corpus, db, client=client)
    with pytest.raises(ValueError, match="not retrieved"):
        ask(db, "pgvector", client=client)


def test_failed_ingestion_preserves_previous_corpus(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    ingest(corpus, db, provider="lexical")
    previous = VectorStore(db).load()[0]
    client = OllamaClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        ingest(corpus, db, client=client)
    assert VectorStore(db).load()[0] == previous


def test_recorded_evaluation_contains_actual_outputs(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    ingest(corpus, db, provider="lexical")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question": "pgvector?", "ground_truth": "Vector search."}))
    output = tmp_path / "records.jsonl"
    samples = record_evaluation(db, questions, output)
    record = json.loads(output.read_text())
    assert record["answer"] == samples[0].answer
    assert record["contexts"] == ["pgvector adds vector search to PostgreSQL."]
    assert record["latency_ms"] >= 0
    assert record["cost_usd"] is None
    assert record["abstained"] is False
    assert record["citations"] == {"1": "database.txt#chunk-0"}


def test_corrupt_dimensions_fail_explicitly(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    client = OllamaClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"embeddings": [[1, 2], [1]]})
        )
    )
    with pytest.raises(ValueError, match="dimensions"):
        ingest(corpus, db, client=client)


@pytest.fixture
def model_response(corpus, tmp_path):
    """Exercise application parsing through the real HTTP provider adapter."""
    response = {}

    def handle(request):
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0] for _ in payload["input"]]})
        assert payload["stream"] is False
        assert payload["format"]["required"] == ["answer", "citations", "abstain"]
        assert payload["format"]["additionalProperties"] is False
        assert payload["options"]["temperature"] == 0
        return httpx.Response(200, json={"message": {"content": response["raw"]}})

    client = OllamaClient(transport=httpx.MockTransport(handle))
    db = tmp_path / "corpus.db"
    ingest(corpus, db, client=client)

    def run(value):
        response["raw"] = value if isinstance(value, str) else json.dumps(value)
        return ask(db, "question", client=client)

    return run


def test_structured_answer_renders_declared_sources(model_response):
    result = model_response({"answer": "  Answer.  ", "citations": [2, 1], "abstain": False})
    assert result.answer == "Answer. [1] [2]"
    assert result.abstained is False
    assert result.citations == {"1": "agents.txt#chunk-0", "2": "database.txt#chunk-0"}


@pytest.mark.parametrize("wording", ["", ABSTENTION, "The supplied document lacks information."])
def test_abstention_uses_explicit_flag_not_exact_wording(model_response, wording):
    result = model_response({"answer": wording, "citations": [], "abstain": True})
    assert result.answer == ABSTENTION
    assert result.abstained is True
    assert result.citations == {}


@pytest.mark.parametrize(
    "change, error",
    [
        ({"citations": []}, "no source citations"),
        ({"citations": [3]}, "not retrieved"),
        ({"citations": [0]}, "greater than 0"),
        ({"citations": [-1]}, "greater than 0"),
        ({"citations": [True]}, "valid integer"),
        ({"citations": [1.0]}, "valid integer"),
        ({"citations": ["1"]}, "valid integer"),
        ({"citations": [1, 1]}, "unique"),
        ({"abstain": "false"}, "valid boolean"),
        ({"abstain": True}, "cannot contain citations"),
        ({"answer": " "}, "requires answer text"),
        ({"answer": ABSTENTION}, "requires answer text"),
        ({"answer": "Answer [2]"}, "not in answer text"),
        ({"unexpected": "value"}, "Extra inputs"),
    ],
)
def test_invalid_structured_response_fails_explicitly(model_response, change, error):
    value = {"answer": "Answer.", "citations": [1], "abstain": False} | change
    with pytest.raises(ValueError, match=error):
        model_response(value)


@pytest.mark.parametrize("missing", ["answer", "citations", "abstain"])
def test_missing_fields_are_not_inferred(model_response, missing):
    value = {"answer": "Answer.", "citations": [1], "abstain": False}
    del value[missing]
    with pytest.raises(ValueError, match="Field required"):
        model_response(value)


@pytest.mark.parametrize("raw", ["Answer [1]", '{"answer":', "```json\n{}\n```", "[]"])
def test_malformed_provider_output_is_rejected(model_response, raw):
    with pytest.raises(ValueError):
        model_response(raw)


def test_lexical_source_markers_do_not_become_citations(tmp_path):
    doc = tmp_path / "facts.txt"
    doc.write_text("Vectors were described in reference [99].")
    db = tmp_path / "db"
    ingest(doc, db, provider="lexical")
    result = ask(db, "Vectors?")
    assert result.citations == {"1": "facts.txt#chunk-0"}


def test_bad_provider_vector_count():
    client = OllamaClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"embeddings": []}))
    )
    with pytest.raises(ValueError, match="number of vectors"):
        client.embed(["text"], "model")


def test_empty_embedding_rejected(corpus, tmp_path):
    client = OllamaClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"embeddings": [[], []]}))
    )
    with pytest.raises(ValueError, match="non-empty"):
        ingest(corpus, tmp_path / "db", client=client)


def test_ingestion_input_boundaries(corpus, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        ingest(corpus, tmp_path / "db", chunk_size=0)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="non-empty"):
        ingest(empty, tmp_path / "db")
    with empty.open("wb") as stream:
        stream.truncate(10_000_001)
    with pytest.raises(ValueError, match="10 MB"):
        ingest(empty, tmp_path / "db")


def test_ignored_files_and_single_file_ingestion(corpus, tmp_path):
    (corpus / "ignore.json").write_text("not indexed")
    (corpus / "link.txt").symlink_to(corpus / "agents.txt")
    db = tmp_path / "db"
    assert ingest(corpus, db, provider="lexical") == 2
    assert ingest(corpus / "agents.txt", db, provider="lexical") == 1
    assert VectorStore(db).load()[1][0].source == "agents.txt"


def test_query_and_evaluation_input_boundaries(corpus, tmp_path):
    db = tmp_path / "db"
    with pytest.raises(ValueError, match="missing"):
        ask(db, "q")
    ingest(corpus, db, provider="lexical")
    for question in [" ", "q" * 8001]:
        with pytest.raises(ValueError, match="characters"):
            ask(db, question)
    with pytest.raises(ValueError, match="top_k"):
        ask(db, "q", top_k=0)
    questions = tmp_path / "empty.jsonl"
    questions.write_text("")
    with pytest.raises(ValueError, match="unique"):
        record_evaluation(db, questions, tmp_path / "records")
