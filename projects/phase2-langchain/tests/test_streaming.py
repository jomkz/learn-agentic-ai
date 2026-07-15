"""Tests for the FastAPI SSE streaming endpoint."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from streaming import _event_generator, app

client = TestClient(app)


async def _async_collect(gen) -> list[str]:
    return [chunk async for chunk in gen]


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stream_endpoint_exists():
    with client.stream("GET", "/stream?query=test") as response:
        assert response.status_code == 200


def test_event_generator_yields_data():
    chunks = asyncio.run(_async_collect(_event_generator("hello")))
    assert len(chunks) > 0
    data_chunks = [c for c in chunks if c.startswith("data:")]
    assert len(data_chunks) > 0


def test_event_generator_ends_with_done():
    chunks = asyncio.run(_async_collect(_event_generator("x")))
    assert chunks[-1] == "data: [DONE]\n\n"
