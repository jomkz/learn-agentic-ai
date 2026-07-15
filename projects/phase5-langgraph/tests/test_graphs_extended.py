"""Extended tests for LangGraph StateGraph node functions."""

from __future__ import annotations

from graphs import draft_node, review_node, route_after_review, search_node


def _research_state(**kwargs) -> dict:
    base = {
        "query": "test query",
        "search_results": [],
        "draft": "",
        "revision_count": 0,
        "approved": False,
    }
    base.update(kwargs)
    return base


def test_search_node_adds_results():
    out = search_node(_research_state(query="test"))
    assert len(out["search_results"]) > 0


def test_draft_node_increments_revision():
    state = _research_state(search_results=["some result"], revision_count=0)
    out = draft_node(state)
    assert out["revision_count"] == 1


def test_review_node_approves_at_two():
    state = _research_state(revision_count=2)
    out = review_node(state)
    assert out["approved"] is True


def test_review_node_not_approved_at_one():
    state = _research_state(revision_count=1)
    out = review_node(state)
    assert out["approved"] is False


def test_route_after_review_approved():
    assert route_after_review({"approved": True}) == "end"


def test_route_after_review_not_approved():
    assert route_after_review({"approved": False}) == "draft"
