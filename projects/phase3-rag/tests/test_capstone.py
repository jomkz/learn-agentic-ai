from __future__ import annotations

from fastapi.testclient import TestClient

from capstone import SAMPLE_DOCS, DocSearchService, app, service


def test_sample_docs_count() -> None:
    assert len(SAMPLE_DOCS) == 5


def test_sample_docs_structure() -> None:
    assert all("title" in d and "content" in d for d in SAMPLE_DOCS)


def test_service_setup_creates_chunks() -> None:
    s = DocSearchService()
    s.setup()
    assert len(s.chunks) > 0


def test_service_ready_after_setup() -> None:
    s = DocSearchService()
    s.setup()
    assert s.ready is True


def test_search_finds_pods() -> None:
    results = service.search("pod lifecycle")
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_no_match() -> None:
    assert service.search("zzzzz_nonexistent_qqqqq") == []


def test_answer_without_llm_returns_string() -> None:
    answer = service.answer("what is a pod")
    assert isinstance(answer, str)


def test_health_endpoint() -> None:
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["ready"] is True


def test_search_endpoint() -> None:
    client = TestClient(app)
    data = client.get("/search?q=pod").json()
    assert data["count"] >= 0


def test_ask_endpoint_has_answer() -> None:
    client = TestClient(app)
    data = client.get("/ask?q=what+is+a+pod").json()
    assert data["answer"] is not None
