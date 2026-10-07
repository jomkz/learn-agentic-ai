"""Unit tests must not contact live providers, regardless of installed extras or API keys."""

import httpx
import pytest


@pytest.fixture(autouse=True)
def prevent_live_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("Live HTTP is disabled in unit tests; supply a MockTransport")

    async def blocked_async(*args, **kwargs):
        raise RuntimeError("Live HTTP is disabled in unit tests; supply a MockTransport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
