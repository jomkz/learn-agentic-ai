"""Extended coverage tests for Phase 1 config properties and defaults."""

from __future__ import annotations

from config import AgentConfig, AnthropicProviderConfig, OllamaProviderConfig, OpenAIProviderConfig


def test_provider_name_ollama() -> None:
    cfg = AgentConfig(name="x", provider=OllamaProviderConfig())
    assert cfg.provider_name == "ollama"


def test_provider_name_openai() -> None:
    cfg = AgentConfig(name="x", provider=OpenAIProviderConfig(api_key="k"))
    assert cfg.provider_name == "openai"


def test_model_name_ollama() -> None:
    cfg = AgentConfig(name="x", provider=OllamaProviderConfig(model="llama3.1:8b"))
    assert cfg.model_name == "llama3.1:8b"


def test_model_name_anthropic() -> None:
    cfg = AgentConfig(
        name="x",
        provider=AnthropicProviderConfig(api_key="k", model="claude-haiku-4-5-20251001"),
    )
    assert cfg.model_name == "claude-haiku-4-5-20251001"


def test_system_prompt_none_default() -> None:
    cfg = AgentConfig(name="x", provider=OllamaProviderConfig())
    assert cfg.system_prompt is None


def test_retry_config_in_agent() -> None:
    cfg = AgentConfig(name="x", provider=OllamaProviderConfig())
    assert cfg.retry.max_attempts == 3
