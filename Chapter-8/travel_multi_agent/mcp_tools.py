"""MCP tool registry with persistent sessions for direct agent binding.

Every MCP server used in this chapter is registered once in ``MCP_SERVERS``.
``load_tools`` keeps one live session per server (stdio processes stay open for
the lifetime of the owning A2A service) instead of spawning a new process per
tool call. Call ``close_all_sessions`` on shutdown.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


def _baidu_key() -> str:
    """Return the Baidu Map API key from whichever supported env var is set."""
    return (
        os.getenv("BAIDU_MAP_API_KEY")
        or os.getenv("BAIDU_MAPS_API_KEY")
        or os.getenv("BAIDU_MAP_AK")
        or ""
    ).strip()


MCP_SERVERS: Dict[str, Dict[str, Any]] = {
    "weather": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "weatherapi-mcp"],
        "env": {
            "WEATHERAPI_KEY": (os.getenv("WEATHERAPI_KEY") or "").strip(),
            "PYTHONUTF8": "1",
        },
    },
    "baidu": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@baidumap/mcp-server-baidu-map"],
        "env": {"BAIDU_MAP_API_KEY": _baidu_key(), "PYTHONUTF8": "1"},
    },
    "variflight": {
        "transport": "streamable_http",
        "url": os.getenv(
            "VARIFLIGHT_MCP_URL",
            "https://ai.variflight.com/servers/aviation/mcp",
        ),
        "headers": {"X-API-Key": (os.getenv("VARIFLIGHT_API_KEY") or "").strip()},
    },
}


class MCPSession:
    """One live MCP session whose stdio/HTTP connection stays open for reuse."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.config = MCP_SERVERS[server_name]
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._session: Any = None
        self._errlog: Any = None
        self._http_client: Any = None
        self.tools: List[BaseTool] = []

    async def start(self) -> List[BaseTool]:
        """Open the transport and ClientSession, then load its LangChain tools."""
        transport = self.config.get("transport")
        if transport == "stdio":
            params = StdioServerParameters(
                command=self.config["command"],
                args=self.config["args"],
                env=self.config.get("env") or None,
            )
            # errlog must have a real file descriptor: inside a Jupyter kernel
            # sys.stderr has no fileno(), which breaks subprocess spawning.
            self._errlog = open(os.devnull, "w")
            self._transport_cm = stdio_client(params, errlog=self._errlog)
            read, write = await self._transport_cm.__aenter__()
        elif transport == "streamable_http":
            import httpx

            headers = self.config.get("headers") or {}
            self._http_client = httpx.AsyncClient(headers=headers)
            self._transport_cm = streamable_http_client(
                self.config["url"], http_client=self._http_client
            )
            read, write, _ = await self._transport_cm.__aenter__()
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        self.tools = await load_mcp_tools(self._session)
        return self.tools

    async def close(self) -> None:
        """Close the session and release the underlying process/connection."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_cm = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        if self._errlog is not None:
            try:
                self._errlog.close()
            except Exception:
                pass
            self._errlog = None
        self._session = None
        self.tools = []


_sessions: Dict[str, MCPSession] = {}


async def load_tools(server_name: str) -> List[BaseTool]:
    """Return the tools of one MCP server, opening and reusing a live session."""
    if server_name not in MCP_SERVERS:
        raise ValueError(f"Unknown MCP server: {server_name}")
    session = _sessions.get(server_name)
    if session is None:
        session = MCPSession(server_name)
        _sessions[server_name] = session
        await session.start()
    return session.tools


async def close_all_sessions() -> None:
    """Close every open MCP session."""
    for session in _sessions.values():
        await session.close()
    _sessions.clear()