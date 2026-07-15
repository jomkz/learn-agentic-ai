"""Tests for LangGraph state graphs and multi-agent supervisor pattern."""

from __future__ import annotations

from graphs import build_research_graph
from multi_agent import build_supervisor_graph


def _initial_research_state() -> dict:
    return {
        "query": "LangGraph checkpointing",
        "search_results": [],
        "draft": "",
        "revision_count": 0,
        "approved": False,
    }


def test_research_graph_produces_draft():
    graph = build_research_graph()
    result = graph.invoke(
        _initial_research_state(),
        config={"configurable": {"thread_id": "test-draft-1"}},
    )
    assert isinstance(result["draft"], str)
    assert len(result["draft"]) > 0


def test_research_graph_approves_after_revisions():
    graph = build_research_graph()
    result = graph.invoke(
        _initial_research_state(),
        config={"configurable": {"thread_id": "test-approve-1"}},
    )
    assert result["approved"] is True


def test_supervisor_routes_to_researcher():
    graph = build_supervisor_graph()
    result = graph.invoke(
        {"messages": [], "next_agent": "", "task": "research quantum computing", "results": []},
    )
    assert any("Research result" in r for r in result["results"])


def test_supervisor_routes_to_writer():
    graph = build_supervisor_graph()
    result = graph.invoke(
        {"messages": [], "next_agent": "", "task": "write a blog post", "results": []},
    )
    assert any("Written summary" in r for r in result["results"])


def test_graph_state_persists_across_invocations():
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "test-persist-2"}}
    result = graph.invoke(_initial_research_state(), config=config)
    # State is persisted in the MemorySaver checkpointer
    saved = graph.get_state(config)
    assert saved.values["revision_count"] == result["revision_count"]
    assert saved.values["approved"] is True
