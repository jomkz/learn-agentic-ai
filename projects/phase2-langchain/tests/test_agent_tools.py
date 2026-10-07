"""Tests for agent tools: search_web, calculate, get_current_time."""

from __future__ import annotations

import pytest

from mobility_ai.phase2.agent import calculate, get_current_time, search_web


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


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__.__mro__[1].__subclasses__()",
        "sqrt.__call__(9)",
        "[x for x in (1,2)]",
        "'x' * 1000000",
        "2 ** 1000000000",
        "1e309",
        "sqrt(x=9)",
        "True + 1",
        "[1][0]",
        "1+" * 300 + "1",
    ],
)
def test_calculator_rejects_non_arithmetic_and_unbounded_work(expression):
    assert calculate.invoke(expression).startswith("Error")


def test_calculator_supported_expression():
    assert float(calculate.invoke("sqrt(9) + 2 * (4 - 1)")) == 9
