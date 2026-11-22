"""Streamable HTTP MCP client for the local hotel tool server.

Main capabilities:

- Connect to the local Streamable HTTP MCP endpoint.
- Discover and call the published hotel recommendation tool.

Reference:
  MCP 2025-11-25 Transports:
  https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from paths import bootstrap_paths
from settings import mcp_url

bootstrap_paths()


def _parse_tool_result(result: CallToolResult) -> Any:
    if result.isError:
        raise RuntimeError(f"MCP 工具错误: {result.content}")

    if result.structuredContent is not None:
        payload = result.structuredContent
        if isinstance(payload, dict) and "result" in payload and len(payload) == 1:
            return payload["result"]
        return payload

    for block in result.content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        text = str(text).strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    raise RuntimeError("MCP 工具未返回可读内容")


async def _call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call one MCP tool over the standard Streamable HTTP transport.

    The SDK handles the MCP endpoint request and optional request-scoped SSE
    response stream defined by MCP 2025-11-25.

    Reference:
      https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
    """
    url = mcp_url()
    async with streamablehttp_client(
        url,
        timeout=10.0,
        sse_read_timeout=300.0,
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _parse_tool_result(result)


async def fetch_hotels_via_mcp(
    city: str,
    *,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Call ``recommend_hotel_tool`` through MCP.

    Reference:
      https://modelcontextprotocol.io/specification/2025-11-25/server/tools
    """
    try:
        payload = await _call_tool(
            "recommend_hotel_tool",
            {
                "city": city,
                "preferences": preferences,
                "budget_cny_per_night_max": budget_cny_per_night_max,
            },
        )
        if isinstance(payload, dict):
            payload.setdefault("data_source", "mcp-streamable-http/recommend_hotel_tool")
            return payload
        return {
            "raw": payload,
            "data_source": "mcp-streamable-http/recommend_hotel_tool",
        }
    except Exception as exc:
        print(f"[hotel-Mcp] 查询失败: {exc}", flush=True)
        print(
            f"[hotel-Mcp] 请确认 MCP Server 已启动且端点可达: {mcp_url()}",
            flush=True,
        )
        return None
