"""酒店 MCP 客户端：通过 HTTP/SSE 连接 hotel_mcp_server.py。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult

from mcp_paths import bootstrap_paths

bootstrap_paths()

_DEFAULT_HOST = os.getenv("HOTEL_MCP_HOST", "127.0.0.1")
_DEFAULT_PORT = os.getenv("HOTEL_MCP_PORT", "8765")
DEFAULT_SSE_URL = os.getenv("HOTEL_MCP_SSE_URL", f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}/sse")


def sse_url() -> str:
    return (os.getenv("HOTEL_MCP_SSE_URL") or DEFAULT_SSE_URL).strip()


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
    url = sse_url()
    async with sse_client(url, timeout=10.0, sse_read_timeout=300.0) as (read, write):
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
    """通过 SSE MCP 调用 recommend_hotel_tool。"""
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
            payload.setdefault("data_source", "hotel-Mcp-sse/recommend_hotel_tool")
            return payload
        return {"raw": payload, "data_source": "hotel-Mcp-sse/recommend_hotel_tool"}
    except Exception as exc:
        print(f"[hotel-Mcp] 查询失败: {exc}", flush=True)
        print(f"[hotel-Mcp] 请先启动服务: python hotel_mcp_server.py  (SSE: {sse_url()})", flush=True)
        return None


async def ask_hotel_agent_via_mcp(user_query: str) -> Optional[str]:
    """通过 SSE MCP 调用 hotel_agent_query（完整 LangChain Agent）。"""
    try:
        payload = await _call_tool("hotel_agent_query", {"user_query": user_query})
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        print(f"[hotel-Mcp] Agent 调用失败: {exc}", flush=True)
        print(f"[hotel-Mcp] 请先启动服务: python hotel_mcp_server.py  (SSE: {sse_url()})", flush=True)
        return None


def fetch_hotels_via_mcp_sync(
    city: str,
    *,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    return asyncio.run(
        fetch_hotels_via_mcp(
            city,
            preferences=preferences,
            budget_cny_per_night_max=budget_cny_per_night_max,
        )
    )


def ask_hotel_agent_via_mcp_sync(user_query: str) -> Optional[str]:
    return asyncio.run(ask_hotel_agent_via_mcp(user_query))


def close_hotel_mcp() -> None:
    """SSE 模式无持久子进程，保留接口以兼容 demo。"""
    return None


if __name__ == "__main__":
    # print(f"[hotel-Mcp] 连接 SSE: {sse_url()}", flush=True)
    # print("[hotel-Mcp] 查询酒店列表...", flush=True)
    # data = fetch_hotels_via_mcp_sync("大同", preferences="近景区")
    # if data:
    #     print(json.dumps(data, ensure_ascii=False, indent=2))
    print("\n[hotel-Mcp] 调用酒店 Agent...", flush=True)
    answer = ask_hotel_agent_via_mcp_sync("我要去大同玩三天，需要近景区，推荐一家酒店")
    if answer:
        print(answer)
