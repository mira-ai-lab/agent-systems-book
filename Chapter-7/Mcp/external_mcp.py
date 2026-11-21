"""Reusable helpers for connecting to external stdio MCP servers."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping, Sequence

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@asynccontextmanager
async def stdio_mcp_tools(
    command: str,
    args: Sequence[str],
    env: Mapping[str, str],
) -> AsyncIterator[list[Any]]:
    """Start a stdio MCP process, initialize it, and yield its LangChain tools.

    Jupyter on Windows cannot use its captured stderr stream as a subprocess
    file descriptor, so the MCP process receives ``os.devnull`` as ``errlog``.
    """
    params = StdioServerParameters(
        command=command,
        args=list(args),
        env=dict(env),
    )
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield await load_mcp_tools(session)


def full_text(value: Any) -> str:
    """Convert a tool payload or LangChain message into complete display text."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)
    return str(content)


async def run_agent_once(
    agent: Any,
    user_query: str,
    thread_id: str,
) -> None:
    """Run an Agent and print complete tool and final outputs without streaming."""
    async for event in agent.astream_events(
        {"messages": [("user", user_query)]},
        {"configurable": {"thread_id": thread_id}},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            print("MCP tool call:", event.get("name"))
            print("Arguments:", full_text(event["data"].get("input")))

        elif kind == "on_tool_end":
            print("Tool returned:")
            print(full_text(event["data"].get("output")))

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            output = event.get("data", {}).get("output") or {}
            messages = output.get("messages", []) if isinstance(output, dict) else []
            if messages:
                print("\\nFinal answer:")
                print(full_text(messages[-1]))
            return
