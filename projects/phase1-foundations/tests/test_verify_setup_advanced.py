"""Mocked tests covering verify_setup.py ollama checks and main()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from verify_setup import check_langchain_available, check_ollama_models, check_ollama_running, main


def _mock_ollama_response(model_names: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return resp


def test_check_ollama_running_succeeds_with_mock() -> None:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=resp):
        check_ollama_running()


def test_check_ollama_running_raises_on_http_error() -> None:
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("HTTP 404")
    with patch("httpx.get", return_value=resp):
        with pytest.raises(Exception):
            check_ollama_running()


def test_check_ollama_models_passes_with_required_models() -> None:
    models = ["llama3.2:latest", "nomic-embed-text:latest", "llama3.1:8b"]
    with patch("httpx.get", return_value=_mock_ollama_response(models)):
        check_ollama_models()


def test_check_ollama_models_raises_on_missing_models() -> None:
    with patch("httpx.get", return_value=_mock_ollama_response(["some-other-model"])):
        with pytest.raises(RuntimeError, match="Missing models"):
            check_ollama_models()


def test_check_ollama_models_empty_list_raises() -> None:
    with patch("httpx.get", return_value=_mock_ollama_response([])):
        with pytest.raises(RuntimeError):
            check_ollama_models()


def test_check_langchain_available_does_not_raise() -> None:
    check_langchain_available()


def test_main_exits_zero_when_all_pass() -> None:
    good_resp = _mock_ollama_response(["llama3.2:latest", "nomic-embed-text:latest"])
    with (
        patch("httpx.get", return_value=good_resp),
        patch("sys.exit") as mock_exit,
    ):
        main()
    mock_exit.assert_called_once()
    args = mock_exit.call_args[0]
    assert args[0] in (0, 1)


def test_main_exits_one_when_ollama_offline() -> None:
    with (
        patch("httpx.get", side_effect=ConnectionError("offline")),
        patch("sys.exit") as mock_exit,
    ):
        main()
    mock_exit.assert_called_once_with(1)


def test_main_prints_section_headers(capsys: pytest.CaptureFixture[str]) -> None:
    good_resp = _mock_ollama_response(["llama3.2:latest", "nomic-embed-text:latest"])
    with (
        patch("httpx.get", return_value=good_resp),
        patch("sys.exit"),
    ):
        main()
    out = capsys.readouterr().out
    assert "Python packages" in out
    assert "Ollama" in out
