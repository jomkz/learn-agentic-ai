from __future__ import annotations

import asyncio
import importlib


def _reload_portable_app():
    import portable_app

    importlib.reload(portable_app)
    return portable_app


def test_unknown_provider_returns_error(monkeypatch):
    monkeypatch.setenv("PROVIDER", "unknown")
    app = _reload_portable_app()
    result = asyncio.run(app.answer_question("test"))
    assert "Unknown provider" in result


def test_llamastack_provider_handles_offline(monkeypatch):
    monkeypatch.setenv("PROVIDER", "llamastack")
    monkeypatch.setenv("LLAMASTACK_BASE_URL", "http://localhost:59999")
    app = _reload_portable_app()
    result = asyncio.run(app.answer_question("hello"))
    assert isinstance(result, str)


def test_llamastack_config_defaults():
    from llamastack_client import LlamaStackConfig

    config = LlamaStackConfig()
    assert "5001" in config.base_url
