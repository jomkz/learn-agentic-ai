"""
Prompt engineering iteration — three progressively refined prompt versions
applied to the same task: extracting action items from a meeting transcript.

Demonstrates how system prompts and few-shot examples close the quality gap
between a vague first attempt and production-ready output.

Usage:
    uv run python projects/phase1-foundations/prompt_engineering.py
"""

from __future__ import annotations

import asyncio
import os
import time

import openai
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT = """
Alex: Alright, let's wrap up. Sarah, can you send the updated deck to the client by EOD Thursday?
Sarah: Sure. I'll also loop in Ben once the slides are done.
Marcus: I need the API credentials from DevOps before I can push the integration.
Alex: Got it — I'll ping DevOps today. Marcus, write up the integration spec for DevOps.
Marcus: Will do. Probably by end of next week.
Alex: Great. Let's reconvene Friday at 10am to check progress.
"""

# ── V1: bare prompt, no guidance ─────────────────────────────────────────────

V1_USER = f"Extract the action items from this transcript:\n\n{TRANSCRIPT}"

# ── V2: system prompt with role + format instruction ─────────────────────────

V2_SYSTEM = (
    "You are a meeting analyst. Given a meeting transcript, extract every action item as a "
    "concise, structured list. For each item include: the owner (who is responsible), the task "
    "(what they will do), and the deadline (when, or 'unspecified'). "
    "Return only the structured list — no preamble or explanation."
)

V2_USER = f"Meeting transcript:\n\n{TRANSCRIPT}"

# ── V3: system prompt + two few-shot examples ─────────────────────────────────

V3_SYSTEM = V2_SYSTEM

V3_USER = f"""Here are two example extractions to calibrate your output format:

---
Example 1 transcript:
"Jane updates the roadmap by Friday. Tom sends the budget to finance before noon tomorrow."

Example 1 output:
- Owner: Jane | Task: Update the roadmap | Deadline: Friday
- Owner: Tom  | Task: Send budget to finance | Deadline: Tomorrow noon
---

Example 2 transcript:
"We need someone to write the test plan. David volunteered. No hard deadline yet."

Example 2 output:
- Owner: David | Task: Write the test plan | Deadline: Unspecified
---

Now extract action items from this transcript:

{TRANSCRIPT}"""


async def call(
    client: openai.AsyncOpenAI,
    model: str,
    system: str | None,
    user: str,
) -> tuple[str, float, int]:
    """Return (response_text, elapsed_seconds, total_tokens)."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    start = time.perf_counter()
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.0,  # deterministic — we're comparing prompt quality, not sampling
    )
    elapsed = time.perf_counter() - start
    text = completion.choices[0].message.content or ""
    tokens = completion.usage.total_tokens if completion.usage else 0
    return text.strip(), elapsed, tokens


def _divider(label: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print("=" * width)


async def run_on_provider(client: openai.AsyncOpenAI, label: str, model: str) -> None:
    print(f"\n{'▓' * 72}")
    print(f"  Provider: {label}  |  Model: {model}")
    print(f"{'▓' * 72}")

    variants = [
        ("V1 — bare prompt (no system prompt, no examples)", None, V1_USER),
        ("V2 — system prompt with role + format instruction", V2_SYSTEM, V2_USER),
        ("V3 — system prompt + two few-shot examples", V3_SYSTEM, V3_USER),
    ]

    # Run all three variants concurrently against this provider
    tasks = [call(client, model, sys, usr) for _, sys, usr in variants]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for (label_v, _, _), result in zip(variants, results):
        _divider(label_v)
        if isinstance(result, Exception):
            print(f"  ERROR: {result}")
        else:
            text, elapsed, tokens = result
            print(text)
            print(f"\n  [{elapsed:.2f}s | {tokens} tokens]")


async def main() -> None:
    print("Task: extract structured action items from a meeting transcript.")
    print("Comparing three prompt versions — V1 (bare) → V2 (system prompt) → V3 (few-shot).\n")

    providers: list[tuple[openai.AsyncOpenAI, str, str]] = [
        (
            openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            "ollama",
            os.getenv("OLLAMA_MODEL", "llama3.2"),
        ),
    ]

    if key := os.getenv("OPENAI_API_KEY"):
        providers.append(
            (
                openai.AsyncOpenAI(api_key=key),
                "openai",
                os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
        )

    for client, label, model in providers:
        try:
            await run_on_provider(client, label, model)
        except Exception as e:  # noqa: BLE001
            print(f"\nSkipping {label}: {e}")

    print(
        "\n\nObservation: V1 varies wildly between runs. "
        "V2 adds consistency via the system prompt. "
        "V3 locks the output format tightly via examples — "
        "the quality gap widens on smaller/local models."
    )


if __name__ == "__main__":
    asyncio.run(main())
