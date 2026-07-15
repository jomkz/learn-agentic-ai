"""MCP server exposing file and query tools for phase 6."""

from __future__ import annotations

import asyncio
import json
import os

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("phase6-tools")


def list_files(path: str) -> str:
    try:
        entries = os.listdir(path)
        return json.dumps(entries)
    except Exception as exc:
        return f"Error listing {path}: {exc}"


def read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(2000)
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def run_query(sql: str) -> str:
    if not sql.strip().upper().startswith("SELECT"):
        return "Only SELECT queries are allowed"
    return f"Query: {sql}\nResults: (requires database connection)"


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_files",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="read_file",
            description="Read contents of a text file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="run_query",
            description="Run a read-only SQL query and return results as CSV",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SELECT SQL query to execute"},
                },
                "required": ["sql"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "list_files":
        result = list_files(arguments.get("path", ""))
    elif name == "read_file":
        result = read_file(arguments.get("path", ""))
    elif name == "run_query":
        result = run_query(arguments.get("sql", ""))
    else:
        result = f"Unknown tool: {name}"
    return [types.TextContent(type="text", text=result)]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
