"""Tests for llamastack_client.py with mocked llama_stack_client module."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch


def _make_mock_llama_stack() -> MagicMock:
    mock_client_class = MagicMock()
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    completion_msg = MagicMock()
    completion_msg.content.text = "LlamaStack answer"
    mock_completion = MagicMock()
    mock_completion.completion_message = completion_msg
    mock_client.inference.chat_completion.return_value = mock_completion

    mock_agent_response = MagicMock()
    mock_agent_response.agent_id = "agent-123"
    mock_client.agents.create.return_value = mock_agent_response

    mock_session = MagicMock()
    mock_session.session_id = "session-456"
    mock_client.agents.sessions.create.return_value = mock_session

    return mock_client_class


def test_chat_completion_with_mocked_client() -> None:

    mock_cls = _make_mock_llama_stack()
    mock_module = MagicMock()
    mock_module.LlamaStackClient = mock_cls

    with patch.dict(sys.modules, {"llama_stack_client": mock_module}):
        import importlib

        import llamastack_client

        importlib.reload(llamastack_client)
        config = llamastack_client.LlamaStackConfig()
        result = asyncio.run(
            llamastack_client.chat_completion(config, [{"role": "user", "content": "hi"}])
        )

    assert result == "LlamaStack answer"


def test_chat_completion_returns_error_string_on_exception() -> None:

    mock_module = MagicMock()
    mock_module.LlamaStackClient.side_effect = RuntimeError("connection refused")

    with patch.dict(sys.modules, {"llama_stack_client": mock_module}):
        import importlib

        import llamastack_client

        importlib.reload(llamastack_client)
        config = llamastack_client.LlamaStackConfig()
        result = asyncio.run(
            llamastack_client.chat_completion(config, [{"role": "user", "content": "hi"}])
        )

    assert result.startswith("[LlamaStack error:")


def test_create_agent_session_with_mocked_client() -> None:
    mock_cls = _make_mock_llama_stack()
    mock_module = MagicMock()
    mock_module.LlamaStackClient = mock_cls

    with patch.dict(sys.modules, {"llama_stack_client": mock_module}):
        import importlib

        import llamastack_client

        importlib.reload(llamastack_client)
        config = llamastack_client.LlamaStackConfig()
        agent_id, session_id = asyncio.run(
            llamastack_client.create_agent_session(config, tools=["web_search"])
        )

    assert agent_id == "agent-123"
    assert session_id == "session-456"
