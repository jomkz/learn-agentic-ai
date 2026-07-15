"""LangGraph StateGraph fundamentals with checkpointing."""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph


class ResearchState(TypedDict):
    query: str
    search_results: list[str]
    draft: str
    revision_count: int
    approved: bool


def search_node(state: ResearchState) -> dict:
    return {"search_results": [f"Result for: {state['query']}"]}


def draft_node(state: ResearchState) -> dict:
    return {
        "draft": f"Draft based on {len(state['search_results'])} results: ...",
        "revision_count": state.get("revision_count", 0) + 1,
    }


def review_node(state: ResearchState) -> dict:
    return {"approved": state.get("revision_count", 0) >= 2}


def route_after_review(state: ResearchState) -> str:
    return "end" if state.get("approved") else "draft"


def build_research_graph() -> CompiledStateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_node)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_edge("search", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", route_after_review, {"end": END, "draft": "draft"})
    graph.set_entry_point("search")
    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    g = build_research_graph()
    result = g.invoke(
        {
            "query": "LangGraph checkpointing",
            "search_results": [],
            "draft": "",
            "revision_count": 0,
            "approved": False,
        },
        config={"configurable": {"thread_id": "demo-1"}},
    )
    print(result)
