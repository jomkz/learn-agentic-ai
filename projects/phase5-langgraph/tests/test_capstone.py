from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from capstone import PipelineResult, app, run_pipeline

_MOCK_STATE = {"results": ["Research result for: research AI trends"], "next_agent": ""}


def _make_mock_graph(state: dict | None = None):
    g = MagicMock()
    g.invoke.return_value = state if state is not None else _MOCK_STATE
    return g


def test_pipeline_result_model():
    result = PipelineResult(task="t", results=["r"], thread_id="1", agent_count=1)
    assert result.task == "t"
    assert result.results == ["r"]
    assert result.thread_id == "1"
    assert result.agent_count == 1


def test_run_pipeline_research_task():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        result = run_pipeline("research quantum computing")
    assert isinstance(result, PipelineResult)


def test_run_pipeline_has_results():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        result = run_pipeline("research AI trends")
    assert isinstance(result.results, list)


def test_run_pipeline_thread_id_auto():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        result = run_pipeline("write a report")
    assert result.thread_id is not None


def test_run_pipeline_custom_thread_id():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        result = run_pipeline("x", thread_id="custom-123")
    assert result.thread_id == "custom-123"


def test_health_endpoint():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"


def test_agents_endpoint():
    client = TestClient(app)
    assert client.get("/agents").json() == ["researcher", "analyst", "writer"]


def test_status_endpoint():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        client = TestClient(app)
        data = client.get("/status?task=research+AI").json()
    assert data["task"] == "research AI"


def test_run_endpoint_streams():
    with patch("capstone.build_supervisor_graph", return_value=_make_mock_graph()):
        client = TestClient(app)
        response = client.get("/run?task=write+summary")
    assert response.status_code == 200
