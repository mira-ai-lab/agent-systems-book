"""Streamable HTTP MCP server exposing one reusable hotel tool.

启动（默认 http://127.0.0.1:8765/mcp）：
  python server.py

环境变量（可选）：
  HOTEL_MCP_HOST=127.0.0.1
  HOTEL_MCP_PORT=8765

Official references:
  MCP 2025-11-25 Streamable HTTP:
  https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
  MCP 2025-11-25 Tools:
  https://modelcontextprotocol.io/specification/2025-11-25/server/tools
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from paths import bootstrap_paths
from settings import server_host, server_port

bootstrap_paths()

# Cross-chapter reuse: Chapter-6/domain_agents.py
from domain_agents import recommend_hotel

logging.basicConfig(level=logging.INFO)
for noisy_logger in ("httpx", "httpx2", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# This tool is stateless, so the standard Streamable HTTP stateless mode is
# sufficient and avoids retaining per-client MCP sessions.
mcp = FastMCP(
    "hotel-recommendation",
    host=server_host(),
    port=server_port(),
    stateless_http=True,
)


@mcp.tool()
async def recommend_hotel_tool(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Expose the Chapter-6 hotel tool as an MCP tool.

    MCP 2025-11-25 requires tool metadata such as name, description and
    ``inputSchema``. ``@mcp.tool`` derives that metadata from this function.

    Reference:
      https://modelcontextprotocol.io/specification/2025-11-25/server/tools
    """
    result = await recommend_hotel.ainvoke(
        {
            "city": city,
            "preferences": preferences,
            "budget_cny_per_night_max": budget_cny_per_night_max,
        }
    )
    return dict(result) if isinstance(result, dict) else {"result": result}


if __name__ == "__main__":
    from settings import mcp_url

    server_url = mcp_url()
    print(f"[hotel-Mcp] Streamable HTTP 服务启动: {server_url}", flush=True)
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        print("\n[hotel-Mcp] 服务已停止", flush=True)
