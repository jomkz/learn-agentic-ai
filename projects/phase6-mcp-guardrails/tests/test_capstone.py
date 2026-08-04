"""Tests for the Phase 6 A2A agent server capstone."""

from __future__ import annotations

from fastapi.testclient import TestClient

from capstone import AgentCard, A2ATask, A2ATaskStore, process_with_budget, app


def test_agent_card_defaults():
    assert AgentCard().name == "Phase6ResearchAgent"


def test_agent_card_has_skills():
    assert len(AgentCard().skills) == 2


def test_a2a_task_default_status():
    assert A2ATask(id="1", input="q").status == "pending"


def test_task_store_create_and_get():
    s = A2ATaskStore()
    t = s.create("hello")
    assert s.get(t.id).input == "hello"


def test_task_store_complete():
    s = A2ATaskStore()
    t = s.create("q")
    s.complete(t.id, "done", 50)
    assert s.get(t.id).status == "complete"


def test_task_store_fail():
    s = A2ATaskStore()
    t = s.create("q")
    s.fail(t.id, "oops")
    assert s.get(t.id).error == "oops"


def test_process_with_budget_ok():
    assert "result" in process_with_budget("what is RAG")


def test_process_with_budget_too_large():
    assert "error" in process_with_budget("q", max_tokens=1)


def test_agent_card_endpoint():
    client = TestClient(app)
    assert client.get("/.well-known/agent.json").json()["name"] == "Phase6ResearchAgent"


def test_health_endpoint():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"


def test_send_task_endpoint():
    client = TestClient(app)
    resp = client.post("/tasks/send", json={"input": "research RAG"})
    assert resp.status_code == 200
    assert "id" in resp.json()


def test_get_task_endpoint():
    client = TestClient(app)
    post_resp = client.post("/tasks/send", json={"input": "explain embeddings"})
    task_id = post_resp.json()["id"]
    get_resp = client.get(f"/tasks/{task_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == task_id


def test_get_task_404():
    client = TestClient(app)
    assert client.get("/tasks/nonexistent-id").status_code == 404
