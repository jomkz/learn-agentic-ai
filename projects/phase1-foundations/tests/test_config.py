"""Tests for the Phase 1 Pydantic v2 agent configuration models."""

import pytest
from config import (
    AgentConfig,
    AnthropicProviderConfig,
    OllamaProviderConfig,
    OpenAIProviderConfig,
    RetryConfig,
)
from pydantic import ValidationError

# ── Provider-specific config construction ────────────────────────────────────


def test_ollama_defaults() -> None:
    cfg = OllamaProviderConfig()
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3.2"
    assert "11434" in cfg.base_url


def test_ollama_custom_model() -> None:
    cfg = OllamaProviderConfig(model="llama3.1:8b")
    assert cfg.model == "llama3.1:8b"


def test_openai_valid() -> None:
    cfg = OpenAIProviderConfig(api_key="sk-test")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"


def test_anthropic_valid() -> None:
    cfg = AnthropicProviderConfig(api_key="sk-ant-test")
    assert cfg.provider == "anthropic"


# ── Provider api_key validators ──────────────────────────────────────────────


def test_openai_empty_api_key_rejected() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        OpenAIProviderConfig(api_key="   ")


def test_anthropic_empty_api_key_rejected() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        AnthropicProviderConfig(api_key="")


# ── AgentConfig construction ─────────────────────────────────────────────────


def test_agent_with_ollama_provider() -> None:
    cfg = AgentConfig(
        name="local-agent",
        provider=OllamaProviderConfig(),
    )
    assert cfg.provider_name == "ollama"
    assert cfg.model_name == "llama3.2"
    assert cfg.temperature == 0.7
    assert isinstance(cfg.retry, RetryConfig)


def test_agent_with_openai_provider() -> None:
    cfg = AgentConfig(
        name="cloud-agent",
        provider=OpenAIProviderConfig(api_key="sk-test"),
        temperature=0.0,
        max_tokens=256,
    )
    assert cfg.provider_name == "openai"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 256


def test_agent_with_anthropic_provider() -> None:
    cfg = AgentConfig(
        name="claude-agent",
        provider=AnthropicProviderConfig(api_key="sk-ant-test"),
        system_prompt="Be concise.",
    )
    assert cfg.provider_name == "anthropic"
    assert cfg.system_prompt == "Be concise."


# ── AgentConfig field validation ─────────────────────────────────────────────


def test_agent_empty_name_rejected() -> None:
    with pytest.raises(ValidationError, match="name"):
        AgentConfig(
            name="   ",
            provider=OllamaProviderConfig(),
        )


def test_temperature_above_max_rejected() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        AgentConfig(
            name="bad-agent",
            provider=OllamaProviderConfig(),
            temperature=2.1,
        )


def test_temperature_below_min_rejected() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        AgentConfig(
            name="bad-agent",
            provider=OllamaProviderConfig(),
            temperature=-0.1,
        )


def test_max_tokens_below_min_rejected() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        AgentConfig(
            name="bad-agent",
            provider=OllamaProviderConfig(),
            max_tokens=0,
        )


# ── Discriminated union — wrong provider literal ──────────────────────────────


def test_unknown_provider_literal_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            name="bad-agent",
            provider={"provider": "cohere", "model": "command-r"},  # type: ignore[arg-type]
        )


def test_provider_dict_wrong_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            name="bad-agent",
            provider="not-a-dict",  # type: ignore[arg-type]
        )


# ── RetryConfig ───────────────────────────────────────────────────────────────


def test_retry_defaults() -> None:
    r = RetryConfig()
    assert r.max_attempts == 3
    assert r.backoff_seconds == 1.0


def test_retry_custom() -> None:
    r = RetryConfig(max_attempts=5, backoff_seconds=2.5)
    assert r.max_attempts == 5


def test_retry_attempts_below_min_rejected() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(max_attempts=0)


def test_retry_attempts_above_max_rejected() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(max_attempts=11)


# ── Serialisation ─────────────────────────────────────────────────────────────


def test_agent_round_trips_json() -> None:
    original = AgentConfig(
        name="round-trip-agent",
        provider=OpenAIProviderConfig(api_key="sk-test", model="gpt-4o"),
        temperature=0.5,
        system_prompt="You are helpful.",
        retry=RetryConfig(max_attempts=2),
    )
    restored = AgentConfig.model_validate_json(original.model_dump_json())
    assert restored.name == original.name
    assert restored.provider_name == "openai"
    assert restored.model_name == "gpt-4o"
    assert restored.retry.max_attempts == 2


def test_schema_includes_discriminator() -> None:
    schema = AgentConfig.model_json_schema()
    assert "provider" in str(schema)
    assert (
        "discriminator" in str(schema).lower() or "oneOf" in str(schema) or "anyOf" in str(schema)
    )
