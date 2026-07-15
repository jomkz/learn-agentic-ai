"""
Async LLM client demonstrating concurrent calls to multiple providers.

Runs two prompts concurrently against Ollama (always) and a cloud provider
(only if the relevant API key is set), then prints a latency comparison table.

Usage:
    uv run python projects/phase1-foundations/async_client.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import openai
from dotenv import load_dotenv

load_dotenv()

PROMPTS = [
    "In one sentence, what is a transformer neural network?",
    "In one sentence, what is the difference between RAG and fine-tuning?",
]


@dataclass
class CompletionResult:
    provider: str
    model: str
    prompt: str
    response: str
    elapsed: float
    tokens_used: int | None


async def complete(
    client: openai.AsyncOpenAI,
    provider_label: str,
    model: str,
    prompt: str,
) -> CompletionResult:
    start = time.perf_counter()
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - start
    message = completion.choices[0].message.content or ""
    tokens = completion.usage.total_tokens if completion.usage else None
    return CompletionResult(
        provider=provider_label,
        model=model,
        prompt=prompt,
        response=message.strip(),
        elapsed=elapsed,
        tokens_used=tokens,
    )


def build_clients() -> list[tuple[openai.AsyncOpenAI, str, str]]:
    """Return (client, label, model) tuples for every configured provider."""
    clients = [
        (
            openai.AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            "ollama",
            os.getenv("OLLAMA_MODEL", "llama3.2"),
        ),
    ]

    if key := os.getenv("OPENAI_API_KEY"):
        clients.append(
            (
                openai.AsyncOpenAI(api_key=key),
                "openai",
                os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
        )

    if key := os.getenv("ANTHROPIC_API_KEY"):
        import anthropic as anthropic_sdk  # noqa: PLC0415

        # Wrap Anthropic in a thin async shim so we can use the same interface.
        # In Phase 2 we'll use the proper LangChain adapters instead.
        clients.append(
            (
                _AnthropicAsOpenAI(anthropic_sdk.AsyncAnthropic(api_key=key)),
                "anthropic",
                os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            )
        )

    return clients


class _AnthropicAsOpenAI:
    """Minimal shim so Anthropic can be called via the same `complete()` signature."""

    def __init__(self, client: object) -> None:
        self._client = client
        self.chat = self

    async def completions_create(
        self, model: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> object:

        response = await self._client.messages.create(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        class _FakeUsage:
            total_tokens = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else None
            )

        class _FakeMessage:
            content = response.content[0].text if response.content else ""

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeCompletion:
            choices = [_FakeChoice()]
            usage = _FakeUsage()

        return _FakeCompletion()

    # Make `client.chat.completions.create(...)` work.
    @property
    def completions(self) -> _AnthropicAsOpenAI:
        return self

    create = completions_create


def _print_table(results: list[CompletionResult]) -> None:
    col_w = {"provider": 12, "model": 24, "elapsed": 10, "tokens": 8, "response": 60}
    header = (
        f"{'Provider':<{col_w['provider']}} "
        f"{'Model':<{col_w['model']}} "
        f"{'Elapsed':>{col_w['elapsed']}} "
        f"{'Tokens':>{col_w['tokens']}} "
        f"{'Response':<{col_w['response']}}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        tokens_str = str(r.tokens_used) if r.tokens_used else "—"
        truncated = r.response[:57] + "..." if len(r.response) > 60 else r.response
        print(
            f"{r.provider:<{col_w['provider']}} "
            f"{r.model:<{col_w['model']}} "
            f"{r.elapsed:>{col_w['elapsed'] - 1}.2f}s "
            f"{tokens_str:>{col_w['tokens']}} "
            f"{truncated:<{col_w['response']}}"
        )
    print(sep)


async def main() -> None:
    clients = build_clients()
    print(f"Providers configured: {[label for _, label, _ in clients]}")
    print(f"Running {len(PROMPTS)} prompts × {len(clients)} providers concurrently...\n")

    for prompt in PROMPTS:
        print(f"Prompt: {prompt!r}")
        tasks = [complete(client, label, model, prompt) for client, label, model in clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                print(f"  ERROR: {r}")
            else:
                valid_results.append(r)
        _print_table(valid_results)
        print()


if __name__ == "__main__":
    asyncio.run(main())
