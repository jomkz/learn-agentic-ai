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
    assert ask(db, "Jupiter moons").answer == ABSTENTION


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
        return httpx.Response(200, json={"message": {"content": "Unsupported citation [99]"}})

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


def test_corrupt_dimensions_fail_explicitly(corpus, tmp_path):
    db = tmp_path / "corpus.db"
    client = OllamaClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"embeddings": [[1, 2], [1]]})
        )
    )
    with pytest.raises(ValueError, match="dimensions"):
        ingest(corpus, db, client=client)


@pytest.mark.parametrize("answer", ["Answer [1]", ABSTENTION, "Uncited answer"])
def test_model_answer_citation_contract(corpus, tmp_path, answer):
    def handle(request):
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0] for _ in payload["input"]]})
        return httpx.Response(200, json={"message": {"content": answer}})

    client = OllamaClient(transport=httpx.MockTransport(handle))
    db = tmp_path / "corpus.db"
    ingest(corpus, db, client=client)
    if answer == "Uncited answer":
        with pytest.raises(ValueError, match="no source citations"):
            ask(db, "question", client=client)
    else:
        assert ask(db, "question", client=client).answer == answer


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
