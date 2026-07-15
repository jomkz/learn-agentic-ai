from __future__ import annotations

import pytest
from cost import TokenBudget, estimate_cost_usd, tier_route


def test_token_budget_fits() -> None:
    assert TokenBudget(100).fits("hello world") is True


def test_token_budget_overflow() -> None:
    assert TokenBudget(1).fits("a " * 1000) is False


def test_tier_route_short_query() -> None:
    assert tier_route("what is rag") == "cheap"


def test_tier_route_complex_query() -> None:
    assert tier_route("analyze and compare the architecture of two systems") == "expensive"


def test_estimate_cost_gpt4o_mini() -> None:
    result = estimate_cost_usd(1000, 500, "gpt-4o-mini")
    expected = (1000 / 1e6) * 0.15 + (500 / 1e6) * 0.60
    assert result == pytest.approx(expected)
