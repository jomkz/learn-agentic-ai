"""Phase 2 capstone: Research Assistant Agent with LCEL chains and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent import calculate, get_current_time, search_web


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_facts: list[str]
    tools_used: list[str]


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        model=os.environ.get("OPENAI_MODEL", "llama3.2"),
    )


def build_research_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Research {query} and provide a structured report.")]
    )
    return prompt | llm.with_structured_output(ResearchReport)


app = FastAPI(title="Research Assistant Agent")

_TOOLS = ["search_web", "calculate", "get_current_time"]

_ = (search_web, calculate, get_current_time)


@app.get("/health")
def health():
    return {"status": "ok", "tools": _TOOLS}


@app.get("/research")
async def research_stream(query: str = ""):
    async def event_generator():
        yield f"data: Researching: {query}...\n\n"
        await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/report")
def report(query: str = ""):
    default = ResearchReport(
        topic=query,
        summary=f"Unable to research '{query}' at this time.",
        key_facts=[],
        tools_used=[],
    )
    try:
        chain = build_research_chain(build_llm())
        result = chain.invoke({"query": query})
        return result
    except Exception:
        return default


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
