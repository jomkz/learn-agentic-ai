"""Portable Q&A app. Set PROVIDER=llamastack or PROVIDER=anthropic (default: llamastack)."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "llamastack")


async def answer_question(question: str) -> str:
    if PROVIDER == "llamastack":
        from llamastack_client import LlamaStackConfig, chat_completion

        config = LlamaStackConfig(
            base_url=os.getenv("LLAMASTACK_BASE_URL", "http://localhost:5001"),
            model_id=os.getenv("LLAMASTACK_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"),
        )
        messages = [{"role": "user", "content": question}]
        return await chat_completion(config, messages)
    elif PROVIDER == "anthropic":
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            response = await client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=512,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text
        except Exception:
            return "[Anthropic error]"
    else:
        return f"[Unknown provider: {PROVIDER}]"


async def main() -> None:
    question = "What is retrieval-augmented generation?"
    print(f"Question: {question}")
    answer = await answer_question(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
