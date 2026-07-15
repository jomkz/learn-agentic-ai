"""Extended tests for portable_app.py using direct module-level patching."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from portable_app import answer_question


def test_answer_with_llamastack_provider():
    with patch("portable_app.PROVIDER", "llamastack"):
        result = asyncio.run(answer_question("hello"))
    assert isinstance(result, str)


def test_answer_with_anthropic_no_key():
    with (
        patch("portable_app.PROVIDER", "anthropic"),
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "invalid-key"}),
    ):
        result = asyncio.run(answer_question("hello"))
    assert result == "[Anthropic error]" or "error" in result.lower()


def test_answer_with_unknown_provider():
    with patch("portable_app.PROVIDER", "bad_provider"):
        result = asyncio.run(answer_question("hello"))
    assert result == "[Unknown provider: bad_provider]"
