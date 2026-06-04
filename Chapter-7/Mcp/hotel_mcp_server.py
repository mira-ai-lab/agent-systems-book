"""HTTP/SSE MCP Server：把酒店查询与 LangChain Agent 暴露为 MCP 工具。

启动（默认 http://127.0.0.1:8765/sse）：
  python hotel_mcp_server.py

环境变量（可选）：
  HOTEL_MCP_HOST=127.0.0.1
  HOTEL_MCP_PORT=8765
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from hotel_core import recommend_hotel_impl

_HOST = os.getenv("HOTEL_MCP_HOST", "127.0.0.1")
_PORT = int(os.getenv("HOTEL_MCP_PORT", "8765"))

mcp = FastMCP("hotel-recommendation", host=_HOST, port=_PORT)


@mcp.tool()
async def recommend_hotel_tool(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查询酒店候选列表（百度/高德 POI），返回 hotels 数组。"""
    return await recommend_hotel_impl(city, preferences, budget_cny_per_night_max)


@mcp.tool()
async def hotel_agent_query(user_query: str) -> str:
    """理解自然语言，由 LangChain Agent 查酒店并推荐一家。"""
    from hotel_tools import run_hotel_agent

    return await run_hotel_agent(user_query, thread_id="mcp_sse")


if __name__ == "__main__":
    sse_url = f"http://{_HOST}:{_PORT}/sse"
    print(f"[hotel-Mcp] SSE 服务启动: {sse_url}", flush=True)
    mcp.run(transport="sse")
#[hotel-Mcp] SSE 服务启动: http://127.0.0.1:8765/sse