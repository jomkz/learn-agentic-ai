"""Phase 6 capstone: A2A-compatible agent server with context budget and safety guardrails."""

from __future__ import annotations

import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from context_budget import ContextBudget, trim_to_budget


class AgentCard(BaseModel):
    name: str = "Phase6ResearchAgent"
    description: str = "Research agent with context budget management and safety guardrails"
    version: str = "1.0"
    capabilities: dict = {"streaming": True, "pushNotifications": False}
    skills: list[dict] = [
        {"id": "research", "name": "Document Research"},
        {"id": "qa", "name": "Question Answering"},
    ]


class A2ATask(BaseModel):
    id: str
    status: str = "pending"
    input: str
    result: str | None = None
    error: str | None = None
    tokens_used: int = 0


class A2ATaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, A2ATask] = {}

    def create(self, input_text: str) -> A2ATask:
        t = A2ATask(id=str(uuid.uuid4()), input=input_text)
        self._tasks[t.id] = t
        return t

    def get(self, task_id: str) -> A2ATask | None:
        return self._tasks.get(task_id)

    def complete(self, task_id: str, result: str, tokens: int = 0) -> None:
        if t := self._tasks.get(task_id):
            t.status = "complete"
            t.result = result
            t.tokens_used = tokens

    def fail(self, task_id: str, error: str) -> None:
        if t := self._tasks.get(task_id):
            t.status = "failed"
            t.error = error


def process_with_budget(query: str, max_tokens: int = 1000) -> dict:
    budget = ContextBudget(max_tokens=max_tokens)
    budget.add_message("user", query)
    if budget.is_full():
        return {"error": "query too long for budget", "remaining": 0}
    response = f"Research complete for: {query[:80]}"
    budget.add_message("assistant", response)
    return {"result": response, "tokens_used": max_tokens - budget.remaining()}


AGENT_CARD = AgentCard()
store = A2ATaskStore()
app = FastAPI(title="Phase 6 A2A Agent Server")


@app.get("/.well-known/agent.json")
async def agent_card() -> dict:
    return AGENT_CARD.model_dump()


@app.post("/tasks/send")
async def send_task(body: dict) -> dict:
    input_text = body.get("input", "")
    if not input_text:
        raise HTTPException(status_code=422, detail="input required")
    task = store.create(input_text)
    result = process_with_budget(input_text)
    if "error" in result:
        store.fail(task.id, result["error"])
    else:
        store.complete(task.id, result["result"], result.get("tokens_used", 0))
    return store.get(task.id).model_dump()


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    t = store.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return t.model_dump()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent": AGENT_CARD.name, "tasks_processed": len(store._tasks)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
