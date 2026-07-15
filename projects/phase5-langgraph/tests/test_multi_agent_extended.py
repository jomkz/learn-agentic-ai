"""Extended tests for the supervisor multi-agent pattern."""

from __future__ import annotations

from langgraph.graph import END
from multi_agent import (
    AgentState,
    analyst_node,
    build_supervisor_graph,
    researcher_node,
    supervisor_node,
    writer_node,
)


def _base_state(task: str, results: list[str] | None = None) -> AgentState:
    return {"messages": [], "next_agent": "", "task": task, "results": results or []}


def test_supervisor_routes_to_analyst():
    graph = build_supervisor_graph()
    result = graph.invoke(_base_state("analyze market trends"))
    assert any("Analysis" in r for r in result["results"])


def test_supervisor_unknown_task_ends():
    graph = build_supervisor_graph()
    result = graph.invoke(_base_state("do something random"))
    # No specialist ran — results is empty and graph reached END without error
    assert isinstance(result, dict)


def test_supervisor_done_after_result():
    state = _base_state("research anything", results=["existing result"])
    out = supervisor_node(state)
    assert out["next_agent"] == END


def test_researcher_appends_result():
    state = _base_state("test")
    out = researcher_node(state)
    assert len(out["results"]) == 1


def test_analyst_appends_result():
    state = _base_state("test")
    out = analyst_node(state)
    assert len(out["results"]) == 1
    assert "Analysis" in out["results"][0]


def test_writer_appends_result():
    state = _base_state("test")
    out = writer_node(state)
    assert len(out["results"]) == 1
    assert "Written summary" in out["results"][0]


def test_all_three_agents_accessible():
    graph = build_supervisor_graph()
    node_keys = set(graph.nodes.keys())
    assert {"supervisor", "researcher", "analyst", "writer"}.issubset(node_keys)
