"""Extended tests for llamastack_client.py (offline / no server required)."""

from __future__ import annotations

import asyncio

from llamastack_client import LlamaStackConfig, chat_completion, create_agent_session


def test_config_model_id_default():
    config = LlamaStackConfig()
    assert "Llama" in config.model_id


def test_config_shield_id_optional():
    config = LlamaStackConfig(base_url="http://x")
    assert config.shield_id is None


def test_chat_completion_offline_returns_string():
    result = asyncio.run(chat_completion(LlamaStackConfig(), [{"role": "user", "content": "hi"}]))
    assert isinstance(result, str)


def test_chat_completion_offline_contains_bracket():
    result = asyncio.run(chat_completion(LlamaStackConfig(), [{"role": "user", "content": "hi"}]))
    assert result.startswith("[")


def test_create_agent_session_offline_returns_tuple():
    result = asyncio.run(create_agent_session(LlamaStackConfig()))
    assert result == ("agent-offline", "session-offline") or (
        isinstance(result, tuple) and len(result) == 2
    )


def test_create_agent_session_with_tools():
    result = asyncio.run(create_agent_session(LlamaStackConfig(), tools=["web_search"]))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
