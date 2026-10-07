from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobility_ai.phase7.capstone import ProviderStatus, QASession, ask, detect_provider


def test_provider_status_model():
    status = ProviderStatus(provider="openai", model="gpt-4o-mini", base_url=None, available=True)
    assert status.provider == "openai"
    assert status.model == "gpt-4o-mini"
    assert status.base_url is None
    assert status.available is True


def test_qa_session_model():
    session = QASession(session_id="1", provider="openai")
    assert session.questions == []


def test_qa_session_add():
    s = QASession(session_id="1", provider="openai")
    s.add("q", "a")
    assert len(s.history()) == 1


def test_qa_session_history_format():
    s = QASession(session_id="1", provider="openai")
    s.add("q", "a")
    assert s.history()[0] == {"q": "q", "a": "a"}


def test_detect_llamastack(monkeypatch):
    monkeypatch.setenv("PROVIDER", "llamastack")
    result = detect_provider()
    assert result.provider == "llamastack"


def test_detect_openai_with_key(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    result = detect_provider()
    assert result.available is True


def test_detect_openai_no_key(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = detect_provider()
    assert result.available is False


def test_detect_unknown_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "cohere")
    result = detect_provider()
    assert result.available is False


def test_ask_unknown_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "badprovider")
    s = QASession(session_id="x", provider="bad")
    result = asyncio.run(ask(s, "q"))
    assert "Unknown provider" in result


@pytest.mark.parametrize("provider", ["openai", "anthropic", "llamastack"])
async def test_provider_response_is_saved_in_session(monkeypatch, provider):
    from mobility_ai.phase7 import capstone

    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    if provider == "openai":
        import openai

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
    elif provider == "anthropic":
        import anthropic

        response = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use"), SimpleNamespace(type="text", text="answer")]
        )
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kwargs: client)
    else:
        monkeypatch.setattr(capstone, "chat_completion", AsyncMock(return_value="answer"))
    assert detect_provider().provider == provider
    session = QASession(session_id="test", provider=provider)
    assert await ask(session, "question") == "answer"
    assert session.history() == [{"q": "question", "a": "answer"}]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_provider_error_is_visible(monkeypatch, provider):
    import anthropic
    import openai

    def fail(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setattr(openai, "AsyncOpenAI", fail)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", fail)
    assert "provider unavailable" in await ask(QASession(session_id="test", provider=provider), "q")
