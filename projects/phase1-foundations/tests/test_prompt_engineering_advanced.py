"""Mocked tests for prompt_engineering.py call(), _divider(), run_on_provider()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from prompt_engineering import _divider, call, run_on_provider


def _mock_client(text: str = "Action items listed.", tokens: int = 20) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = text
    completion.usage.total_tokens = tokens
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


def test_call_without_system_prompt() -> None:
    client = _mock_client("Answer text")
    text, elapsed, tokens = asyncio.run(call(client, "model", None, "user message"))
    assert text == "Answer text"
    assert elapsed >= 0
    assert tokens == 20


def test_call_with_system_prompt() -> None:
    client = _mock_client("Structured answer")
    text, elapsed, tokens = asyncio.run(call(client, "model", "Be precise.", "user message"))
    assert text == "Structured answer"


def test_call_strips_whitespace() -> None:
    client = _mock_client("  trimmed  ")
    text, _, _ = asyncio.run(call(client, "model", None, "prompt"))
    assert text == "trimmed"


def test_call_handles_none_content() -> None:
    completion = MagicMock()
    completion.choices[0].message.content = None
    completion.usage.total_tokens = 5
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    text, _, _ = asyncio.run(call(client, "model", None, "prompt"))
    assert text == ""


def test_call_returns_elapsed_seconds() -> None:
    client = _mock_client()
    _, elapsed, _ = asyncio.run(call(client, "model", None, "p"))
    assert isinstance(elapsed, float)


def test_divider_prints_label(capsys: pytest.CaptureFixture[str]) -> None:
    _divider("Test Label")
    out = capsys.readouterr().out
    assert "Test Label" in out
    assert "=" in out


def test_divider_prints_separator_lines(capsys: pytest.CaptureFixture[str]) -> None:
    _divider("X")
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) >= 3


async def _run_provider(client: MagicMock) -> None:
    await run_on_provider(client, "test-label", "test-model")


def test_run_on_provider_calls_all_three_variants(capsys: pytest.CaptureFixture[str]) -> None:
    client = _mock_client("- Owner: Alex | Task: Send deck | Deadline: Thursday")
    asyncio.run(_run_provider(client))
    out = capsys.readouterr().out
    assert "V1" in out
    assert "V2" in out
    assert "V3" in out


def test_run_on_provider_shows_provider_label(capsys: pytest.CaptureFixture[str]) -> None:
    client = _mock_client()
    asyncio.run(_run_provider(client))
    out = capsys.readouterr().out
    assert "test-label" in out


def test_run_on_provider_handles_exception(capsys: pytest.CaptureFixture[str]) -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("network error"))
    asyncio.run(_run_provider(client))
    out = capsys.readouterr().out
    assert "ERROR" in out
