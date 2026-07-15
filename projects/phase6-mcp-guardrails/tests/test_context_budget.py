"""Tests for context_budget module."""

from __future__ import annotations

from context_budget import ContextBudget, trim_to_budget


def test_add_and_remaining():
    cb = ContextBudget(max_tokens=1000)
    before = cb.remaining()
    cb.add_message("user", "Hello world")
    assert cb.remaining() < before


def test_trim_oldest_removes_message():
    cb = ContextBudget(max_tokens=1000)
    cb.add_message("user", "First message")
    cb.add_message("assistant", "Second message")
    cb.add_message("user", "Third message")
    count_before = len(cb.get_messages())
    cb.trim_oldest()
    assert len(cb.get_messages()) == count_before - 1


def test_trim_oldest_preserves_system():
    cb = ContextBudget(max_tokens=1000)
    cb.add_message("system", "You are a helpful assistant.")
    cb.add_message("user", "Hello")
    cb.add_message("assistant", "Hi there!")
    cb.trim_oldest()
    msgs = cb.get_messages()
    roles = [m["role"] for m in msgs]
    assert "system" in roles


def test_is_full_when_over_budget():
    cb = ContextBudget(max_tokens=10)
    cb.add_message("user", "This is a sufficiently long message to exceed a tiny token budget")
    assert cb.is_full()


def test_trim_to_budget_keeps_recent():
    messages = [
        {"role": "user", "content": "First old message that takes up space"},
        {"role": "assistant", "content": "Old reply that also takes up space"},
        {"role": "user", "content": "Recent question"},
    ]
    result = trim_to_budget(messages, max_tokens=20)
    contents = [m["content"] for m in result]
    assert "Recent question" in contents
