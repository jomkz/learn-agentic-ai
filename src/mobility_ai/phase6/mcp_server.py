"""MCP server exposing file and query tools for phase 6."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from itertools import islice
from pathlib import Path

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("phase6-tools")


def _open_allowed(path: str, *, root: Path | None = None, directory: bool = False) -> int:
    """Open beneath the trusted root; reject symlinks at every path component."""
    allowed = (root or Path(os.getenv("MCP_ALLOWED_ROOT", "data/mcp"))).resolve(strict=True)
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.relative_to(allowed)
    if ".." in candidate.parts:
        raise ValueError("Parent traversal is not allowed")
    parts = candidate.parts
    fd = os.open(allowed, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1 or directory:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        mode = os.fstat(fd).st_mode
        if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
            raise ValueError("Expected a directory" if directory else "Expected a regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def list_files(path: str, *, root: Path | None = None) -> str:
    try:
        fd = _open_allowed(path, root=root, directory=True)
        try:
            with os.scandir(fd) as entries:
                return json.dumps(sorted(entry.name for entry in islice(entries, 1000)))
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        return f"Error listing file: {exc}"


def read_file(path: str, *, root: Path | None = None) -> str:
    try:
        fd = _open_allowed(path, root=root)
        try:
            return os.read(fd, 2000).decode("utf-8", errors="replace")
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        return f"Error reading file: {exc}"


def run_query(sql: str) -> str:
    if not sql.strip().upper().startswith("SELECT"):
        return "Only SELECT queries are allowed"
    return f"Query: {sql}\nResults: (requires database connection)"


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_files",
            description="List up to 1000 entries beneath MCP_ALLOWED_ROOT; symlinks are forbidden",
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
            description="Read up to 2000 bytes beneath MCP_ALLOWED_ROOT; symlinks are forbidden",
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
            description="Demonstration only: display a SELECT query without executing SQL",
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
