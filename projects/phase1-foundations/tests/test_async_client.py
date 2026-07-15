"""Tests for async_client.py — CompletionResult, build_clients, _print_table, _AnthropicAsOpenAI."""

from __future__ import annotations

from async_client import CompletionResult, _AnthropicAsOpenAI, _print_table, build_clients


def _make_result(**overrides: object) -> CompletionResult:
    defaults: dict[str, object] = {
        "provider": "ollama",
        "model": "llama3.2",
        "prompt": "Hello",
        "response": "Hi there",
        "elapsed": 0.5,
        "tokens_used": 20,
    }
    defaults.update(overrides)
    return CompletionResult(**defaults)  # type: ignore[arg-type]


# ── CompletionResult ──────────────────────────────────────────────────────────


def test_completion_result_is_dataclass() -> None:
    r = CompletionResult("p", "m", "q", "r", 0.1, 10)
    assert r.provider == "p"
    assert r.model == "m"
    assert r.prompt == "q"
    assert r.response == "r"
    assert r.elapsed == 0.1
    assert r.tokens_used == 10


def test_completion_result_nullable_tokens() -> None:
    r = CompletionResult("p", "m", "q", "r", 0.1, None)
    assert r.tokens_used is None


# ── build_clients ─────────────────────────────────────────────────────────────


def test_build_clients_always_has_ollama(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clients = build_clients()
    labels = [label for _, label, _ in clients]
    assert labels[0] == "ollama"


def test_build_clients_no_openai_without_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clients = build_clients()
    labels = [label for _, label, _ in clients]
    assert "openai" not in labels


def test_build_clients_adds_openai_with_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clients = build_clients()
    labels = [label for _, label, _ in clients]
    assert "openai" in labels


# ── _print_table ──────────────────────────────────────────────────────────────


def test_print_table_runs_without_error(capsys) -> None:  # type: ignore[no-untyped-def]
    _print_table([_make_result()])
    captured = capsys.readouterr()
    assert captured.out


def test_print_table_with_long_response(capsys) -> None:  # type: ignore[no-untyped-def]
    long_response = "A" * 80
    _print_table([_make_result(response=long_response)])
    captured = capsys.readouterr()
    assert "..." in captured.out
    assert long_response not in captured.out


def test_print_table_with_none_tokens(capsys) -> None:  # type: ignore[no-untyped-def]
    _print_table([_make_result(tokens_used=None)])
    captured = capsys.readouterr()
    assert "—" in captured.out


# ── _AnthropicAsOpenAI ────────────────────────────────────────────────────────


def test_anthropic_shim_has_completions_attr() -> None:
    shim = _AnthropicAsOpenAI(object())
    assert shim.completions is not None
