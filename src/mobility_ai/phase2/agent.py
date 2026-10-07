"""Tool-calling agent with search, calculator, and time tools."""

from __future__ import annotations

import datetime

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from mobility_ai.phase2.arithmetic import evaluate_arithmetic


@tool
def search_web(query: str) -> str:
    """Search the web for information about a given query."""
    return f"Search results for '{query}': [simulated result about {query}]"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = evaluate_arithmetic(expression)
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
        schema = t.args
        print(f"Tool: {t.name}")
        print(f"  Description: {t.description}")
        print(f"  Schema: {schema}")
        print()
