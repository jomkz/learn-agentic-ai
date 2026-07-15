"""Tests for dspy_rag.py compile_with_bootstrap with mocked dspy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dspy_rag import _DSPY_AVAILABLE, RAGModule, compile_with_bootstrap


def test_compile_passthrough_when_dspy_absent() -> None:
    if _DSPY_AVAILABLE:
        return
    module = RAGModule()
    result = compile_with_bootstrap(module, [], None)
    assert result is module


def test_compile_with_mock_dspy_optimizer() -> None:
    mock_dspy = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer.compile.return_value = "compiled_result"
    mock_dspy.BootstrapFewShot.return_value = mock_optimizer

    with patch("dspy_rag._DSPY_AVAILABLE", True), patch("dspy_rag.dspy", mock_dspy):
        result = compile_with_bootstrap(RAGModule(), [{"q": "x"}], lambda x: True)

    assert result == "compiled_result"
    mock_dspy.BootstrapFewShot.assert_called_once()
    mock_optimizer.compile.assert_called_once()


def test_compile_creates_bootstrap_with_max_demos() -> None:
    mock_dspy = MagicMock()
    mock_dspy.BootstrapFewShot.return_value.compile.return_value = "ok"

    with patch("dspy_rag._DSPY_AVAILABLE", True), patch("dspy_rag.dspy", mock_dspy):
        compile_with_bootstrap("module", [], None)

    call_kwargs = mock_dspy.BootstrapFewShot.call_args[1]
    assert call_kwargs.get("max_bootstrapped_demos") == 4
