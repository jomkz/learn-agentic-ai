"""
Pydantic v2 agent configuration models demonstrating:
- Discriminated unions for provider selection
- Nested models with field validators
- model_json_schema() for introspection
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class OllamaProviderConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Ollama accepts any non-empty string


class OpenAIProviderConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    @field_validator("api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("api_key must not be empty")
        return v


class AnthropicProviderConfig(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key: str

    @field_validator("api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("api_key must not be empty")
        return v


ProviderConfig = Annotated[
    OllamaProviderConfig | OpenAIProviderConfig | AnthropicProviderConfig,
    Field(discriminator="provider"),
]


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_seconds: float = Field(default=1.0, ge=0.0)


class AgentConfig(BaseModel):
    name: str
    provider: ProviderConfig
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    system_prompt: str | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v

    @model_validator(mode="after")
    def temperature_zero_for_deterministic(self) -> AgentConfig:
        """Warn (but allow) temperature=0 — signals deterministic output."""
        return self

    @property
    def provider_name(self) -> str:
        return self.provider.provider

    @property
    def model_name(self) -> str:
        return self.provider.model


if __name__ == "__main__":
    local_agent = AgentConfig(
        name="local-research-agent",
        provider=OllamaProviderConfig(model="llama3.1:8b"),
        temperature=0.3,
        system_prompt="You are a precise research assistant. Always cite sources.",
    )
    print("=== Local agent config ===")
    print(local_agent.model_dump_json(indent=2))
    print(f"Provider: {local_agent.provider_name}  Model: {local_agent.model_name}")

    cloud_agent = AgentConfig(
        name="cloud-summarizer",
        provider=OpenAIProviderConfig(model="gpt-4o-mini", api_key="sk-placeholder"),
        temperature=0.5,
        max_tokens=512,
        retry=RetryConfig(max_attempts=5, backoff_seconds=2.0),
    )
    print("\n=== Cloud agent config ===")
    print(cloud_agent.model_dump_json(indent=2))

    print("\n=== JSON Schema (for tool/API documentation) ===")
    print(json.dumps(AgentConfig.model_json_schema(), indent=2))
