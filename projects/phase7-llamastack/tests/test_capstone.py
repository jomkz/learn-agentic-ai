from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capstone import ProviderStatus, QASession, ask, detect_provider


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
