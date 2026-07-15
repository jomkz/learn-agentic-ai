"""Supervisor multi-agent pattern using LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    task: str
    results: list[str]


AGENTS = ["researcher", "analyst", "writer"]


def supervisor_node(state: AgentState) -> dict:
    if state.get("results"):  # specialist already ran — done
        return {"next_agent": END}
    task = state["task"].lower()
    if "research" in task:
        chosen = "researcher"
    elif "analyz" in task:
        chosen = "analyst"
    elif "writ" in task:
        chosen = "writer"
    else:
        chosen = END
    return {"next_agent": chosen}


def researcher_node(state: AgentState) -> dict:
    return {"results": state.get("results", []) + [f"Research result for: {state['task']}"]}


def analyst_node(state: AgentState) -> dict:
    return {"results": state.get("results", []) + [f"Analysis of: {state['task']}"]}


def writer_node(state: AgentState) -> dict:
    return {"results": state.get("results", []) + [f"Written summary: {state['task']}"]}


def route_from_supervisor(state: AgentState) -> str:
    return state["next_agent"]


def build_supervisor_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()


if __name__ == "__main__":
    g = build_supervisor_graph()
    for task in [
        "research quantum computing",
        "analyze the market trends",
        "write a blog post about AI",
    ]:
        result = g.invoke({"messages": [], "next_agent": "", "task": task, "results": []})
        print(f"Task: {task!r}")
        print(f"Results: {result['results']}\n")
