"""Mocked tests for async_client.py — complete(), Anthropic shim, and build_clients."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from async_client import CompletionResult, _AnthropicAsOpenAI, build_clients, complete


def _mock_openai_completion(text: str = "Test response", tokens: int = 42) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = text
    mock.usage.total_tokens = tokens
    return mock


async def _run_complete_with_mock(text: str = "Mock answer", tokens: int = 10) -> CompletionResult:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_mock_openai_completion(text, tokens))
    return await complete(client, "mock-provider", "mock-model", "test prompt")


def test_complete_returns_completion_result() -> None:
    result = asyncio.run(_run_complete_with_mock())
    assert isinstance(result, CompletionResult)


def test_complete_captures_response_text() -> None:
    result = asyncio.run(_run_complete_with_mock("Hello world"))
    assert result.response == "Hello world"


def test_complete_captures_token_count() -> None:
    result = asyncio.run(_run_complete_with_mock(tokens=99))
    assert result.tokens_used == 99


def test_complete_records_provider_and_model() -> None:
    async def _run() -> CompletionResult:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_mock_openai_completion())
        return await complete(client, "my-provider", "my-model", "q")

    result = asyncio.run(_run())
    assert result.provider == "my-provider"
    assert result.model == "my-model"
    assert result.prompt == "q"


def test_complete_elapsed_is_non_negative() -> None:
    result = asyncio.run(_run_complete_with_mock())
    assert result.elapsed >= 0


def test_complete_handles_empty_content() -> None:
    async def _run() -> CompletionResult:
        client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = None
        mock_completion.usage = None
        client.chat.completions.create = AsyncMock(return_value=mock_completion)
        return await complete(client, "p", "m", "q")

    result = asyncio.run(_run())
    assert result.response == ""
    assert result.tokens_used is None


def test_build_clients_adds_anthropic_with_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clients = build_clients()
    labels = [label for _, label, _ in clients]
    assert "anthropic" in labels


def test_anthropic_shim_completions_property() -> None:
    shim = _AnthropicAsOpenAI(MagicMock())
    assert shim.completions is shim


def test_anthropic_shim_create_is_callable() -> None:
    shim = _AnthropicAsOpenAI(MagicMock())
    assert callable(shim.create)


async def _run_anthropic_completions_create() -> object:
    mock_inner = MagicMock()
    mock_response = MagicMock()
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 10
    mock_response.content[0].text = "Anthropic answer"
    mock_inner.messages.create = AsyncMock(return_value=mock_response)

    shim = _AnthropicAsOpenAI(mock_inner)
    return await shim.completions_create(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.0,
    )


def test_anthropic_shim_completions_create_returns_completion() -> None:
    result = asyncio.run(_run_anthropic_completions_create())
    assert hasattr(result, "choices")
    assert hasattr(result, "usage")


def test_anthropic_shim_completions_create_text_content() -> None:
    result = asyncio.run(_run_anthropic_completions_create())
    assert result.choices[0].message.content == "Anthropic answer"


def test_anthropic_shim_completions_create_token_sum() -> None:
    result = asyncio.run(_run_anthropic_completions_create())
    assert result.usage.total_tokens == 15
