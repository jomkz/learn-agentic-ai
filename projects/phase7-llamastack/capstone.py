"""Phase 7 capstone: Provider-portable Q&A with session history. Set PROVIDER=llamastack|openai|anthropic."""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel

from llamastack_client import LlamaStackConfig, chat_completion

load_dotenv()


class ProviderStatus(BaseModel):
    provider: str
    model: str
    base_url: str | None
    available: bool
    note: str = ""


class QASession(BaseModel):
    session_id: str
    provider: str
    questions: list[str] = []
    answers: list[str] = []

    def add(self, question: str, answer: str) -> None:
        self.questions.append(question)
        self.answers.append(answer)

    def history(self) -> list[dict]:
        return [{"q": q, "a": a} for q, a in zip(self.questions, self.answers)]


def detect_provider() -> ProviderStatus:
    provider = os.getenv("PROVIDER", "llamastack")
    if provider == "llamastack":
        return ProviderStatus(
            provider="llamastack",
            model=os.getenv("LLAMASTACK_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"),
            base_url=os.getenv("LLAMASTACK_BASE_URL", "http://localhost:5001"),
            available=True,
            note="Switch with PROVIDER=openai or PROVIDER=anthropic",
        )
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        return ProviderStatus(
            provider="openai",
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=None,
            available=bool(key),
            note="" if key else "Set OPENAI_API_KEY",
        )
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        return ProviderStatus(
            provider="anthropic",
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            base_url=None,
            available=bool(key),
            note="" if key else "Set ANTHROPIC_API_KEY",
        )
    return ProviderStatus(
        provider=provider,
        model="unknown",
        base_url=None,
        available=False,
        note=f"Unknown provider: {provider}",
    )


async def ask(session: QASession, question: str) -> str:
    provider = os.getenv("PROVIDER", "llamastack")
    if provider == "llamastack":
        config = LlamaStackConfig(
            base_url=os.getenv("LLAMASTACK_BASE_URL", "http://localhost:5001"),
            model_id=os.getenv("LLAMASTACK_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"),
        )
        answer = await chat_completion(config, [{"role": "user", "content": question}])
    elif provider == "openai":
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": question}],
                max_tokens=512,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"[OpenAI error: {e}]"
    elif provider == "anthropic":
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            resp = await client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=512,
                messages=[{"role": "user", "content": question}],
            )
            answer = resp.content[0].text
        except Exception as e:
            answer = f"[Anthropic error: {e}]"
    else:
        answer = f"[Unknown provider: {provider}]"
    session.add(question, answer)
    return answer


async def main() -> None:
    status = detect_provider()
    print(f"Provider: {status.provider} | Model: {status.model}")
    if status.note:
        print(f"Note: {status.note}")
    session = QASession(session_id=str(uuid.uuid4()), provider=status.provider)
    for q in [
        "What is retrieval-augmented generation?",
        "How does LlamaStack differ from LangChain?",
    ]:
        print(f"\nQ: {q}")
        a = await ask(session, q)
        print(f"A: {a[:200]}")
    print(f"\nSession: {len(session.history())} Q&A pairs logged")


if __name__ == "__main__":
    asyncio.run(main())
