from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from capstone import ResearchReport, app, build_llm


def test_research_report_model():
    report = ResearchReport(topic="AI", summary="x", key_facts=["f"], tools_used=["search"])
    assert report.topic == "AI"
    assert report.summary == "x"
    assert report.key_facts == ["f"]
    assert report.tools_used == ["search"]


def test_research_report_defaults():
    report = ResearchReport(topic="x", summary="y", key_facts=[], tools_used=[])
    assert report.tools_used == []


def test_build_llm_returns_model():
    llm = build_llm()
    assert llm is not None


def test_health_endpoint():
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_health_has_tools():
    client = TestClient(app)
    assert "tools" in client.get("/health").json()


def test_research_stream_returns_200():
    client = TestClient(app)
    assert client.get("/research?query=AI").status_code == 200


def test_report_endpoint_handles_offline():
    expected = ResearchReport(
        topic="AI",
        summary="AI is...",
        key_facts=["fact1"],
        tools_used=["search_web"],
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()

    with patch("capstone.build_llm", return_value=mock_llm), patch(
        "capstone.build_research_chain", return_value=mock_chain
    ):
        client = TestClient(app)
        response = client.get("/report?query=AI")

    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "AI"
