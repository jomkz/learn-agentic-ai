"""FastAPI SSE streaming endpoint backed by an LCEL chain."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

_model = FakeListChatModel(responses=["This is a streamed response from the language model."])
_prompt = ChatPromptTemplate.from_template("Answer the following question: {query}")
_chain = _prompt | _model | StrOutputParser()


async def _event_generator(query: str):
    async for chunk in _chain.astream({"query": query}):
        if chunk:
            yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/stream")
async def stream(query: str = "Hello"):
    return StreamingResponse(_event_generator(query), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
