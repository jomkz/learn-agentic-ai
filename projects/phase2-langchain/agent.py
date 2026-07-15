"""Tool-calling agent with search, calculator, and time tools."""

from __future__ import annotations

import datetime
import math

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def search_web(query: str) -> str:
    """Search the web for information about a given query."""
    return f"Search results for '{query}': [simulated result about {query}]"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def get_current_time() -> str:
    """Return the current UTC date and time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def build_agent(model):
    tools = [search_web, calculate, get_current_time]
    return create_react_agent(model, tools)


if __name__ == "__main__":
    tools = [search_web, calculate, get_current_time]
    for t in tools:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        print(f"Tool: {t.name}")
        print(f"  Description: {t.description}")
        print(f"  Schema: {schema}")
        print()
