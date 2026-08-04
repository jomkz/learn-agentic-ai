"""Phase 5 capstone: Autonomous Research Pipeline using LangGraph supervisor and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graphs import ResearchState, build_research_graph
from multi_agent import AgentState, build_supervisor_graph


class PipelineResult(BaseModel):
    task: str
    results: list[str]
    thread_id: str
    agent_count: int


def run_pipeline(task: str, thread_id: str | None = None) -> PipelineResult:
    tid = thread_id or str(uuid.uuid4())
    graph = build_supervisor_graph()
    state = graph.invoke({"messages": [], "next_agent": "", "task": task, "results": []})
    results = state.get("results", [])
    return PipelineResult(
        task=task,
        results=results,
        thread_id=tid,
        agent_count=len(results),
    )


app = FastAPI(title="Autonomous Research Pipeline")


async def _stream_pipeline(task: str):
    result = run_pipeline(task)
    for step in result.results:
        yield f"data: {json.dumps({'step': step})}\n\n"
        await asyncio.sleep(0)
    yield f"data: {json.dumps({'done': True, 'thread_id': result.thread_id})}\n\n"


@app.get("/run")
async def run_endpoint(task: str):
    return StreamingResponse(_stream_pipeline(task), media_type="text/event-stream")


@app.get("/status")
async def status_endpoint(task: str):
    return run_pipeline(task).model_dump()


@app.get("/agents")
async def agents_endpoint():
    return ["researcher", "analyst", "writer"]


@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "graph": "supervisor-multi-agent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
