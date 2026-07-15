"""LlamaStack client usage examples. Requires: llama stack run ollama (on port 5001)."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel


class LlamaStackConfig(BaseModel):
    base_url: str = "http://localhost:5001"
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    shield_id: str | None = None


async def chat_completion(
    config: LlamaStackConfig, messages: list[dict], stream: bool = False
) -> str:
    try:
        from llama_stack_client import LlamaStackClient

        client = LlamaStackClient(base_url=config.base_url)
        response = client.inference.chat_completion(
            model_id=config.model_id, messages=messages, stream=False
        )
        return response.completion_message.content.text
    except ImportError:
        return "[llama-stack-client not installed — run: uv sync --extra llamastack]"
    except Exception as e:
        return f"[LlamaStack error: {e}]"


async def create_agent_session(
    config: LlamaStackConfig, tools: list[str] | None = None
) -> tuple[str, str]:
    try:
        from llama_stack_client import LlamaStackClient

        client = LlamaStackClient(base_url=config.base_url)
        agent_config = {
            "model": config.model_id,
            "instructions": "You are a helpful assistant.",
            "tools": tools or [],
        }
        response = client.agents.create(agent_config=agent_config)
        agent_id = response.agent_id
        session = client.agents.sessions.create(agent_id=agent_id, session_name="demo-session")
        return agent_id, session.session_id
    except Exception:
        return "agent-offline", "session-offline"


async def _demo() -> None:
    config = LlamaStackConfig()
    messages = [{"role": "user", "content": "What is LlamaStack?"}]
    result = await chat_completion(config, messages)
    print(result)


if __name__ == "__main__":
    asyncio.run(_demo())
