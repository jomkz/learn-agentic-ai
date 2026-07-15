"""Tests for agent tools: search_web, calculate, get_current_time."""

from __future__ import annotations

from agent import calculate, get_current_time, search_web


def test_search_web_returns_string():
    result = search_web.invoke("AI")
    assert isinstance(result, str)


def test_search_web_contains_query():
    result = search_web.invoke("AI")
    assert "AI" in result


def test_calculate_simple():
    assert calculate.invoke("2 + 2") == "4"


def test_calculate_sqrt():
    # allowed_names exposes math functions directly, not via the math module object
    result = calculate.invoke("sqrt(9)")
    assert "3" in result


def test_calculate_invalid_graceful():
    result = calculate.invoke("__import__('os').system('ls')")
    assert "Error" in result


def test_get_current_time_is_iso():
    result = get_current_time.invoke({})
    assert "T" in result or "Z" in result


def test_tool_descriptions_non_empty():
    for t in [search_web, calculate, get_current_time]:
        assert t.description and len(t.description.strip()) > 0


def test_tool_names():
    names = {t.name for t in [search_web, calculate, get_current_time]}
    assert names == {"search_web", "calculate", "get_current_time"}
