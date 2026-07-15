"""Tests for Phase 1 environment verification helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from verify_setup import (
    _check,
    check_dotenv,
    check_httpx,
    check_jupyterlab,
    check_ollama_running,
    check_openai_sdk,
    check_pydantic_v2,
)


def test_check_pydantic_v2_passes() -> None:
    check_pydantic_v2()


def test_check_openai_sdk_passes() -> None:
    check_openai_sdk()


def test_check_httpx_passes() -> None:
    check_httpx()


def test_check_dotenv_passes() -> None:
    check_dotenv()


def test_check_jupyterlab_passes() -> None:
    check_jupyterlab()


def test_check_fn_returns_true_on_success() -> None:
    assert _check("test", lambda: None) is True


def test_check_fn_returns_false_on_exception() -> None:
    assert _check("test", lambda: 1 / 0) is False


def test_check_fn_prints_pass(capsys: pytest.CaptureFixture[str]) -> None:
    _check("x", lambda: None)
    assert "PASS" in capsys.readouterr().out


def test_check_fn_prints_fail(capsys: pytest.CaptureFixture[str]) -> None:
    def raiser() -> None:
        raise ValueError("bad")

    _check("x", raiser)
    assert "FAIL" in capsys.readouterr().out


def test_check_ollama_running_fails_offline() -> None:
    with patch("httpx.get", side_effect=ConnectionError("offline")):
        with pytest.raises(Exception):
            check_ollama_running()
