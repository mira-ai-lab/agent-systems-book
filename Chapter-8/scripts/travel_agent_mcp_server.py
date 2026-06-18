"""stdio / SSE MCP Server：把 Chapter-8 LangGraph 旅行多智能体暴露给 Claude Code。

依赖：
    cd Chapter-8
    pip install -e .
    pip install mcp

Claude Code 注册（stdio，推荐）：
    cd Chapter-8
    claude mcp add travel-agent --scope project -- python scripts/travel_agent_mcp_server.py

或项目根 .mcp.json：
    {
      "mcpServers": {
        "travel-agent": {
          "command": "python",
          "args": ["Chapter-8/scripts/travel_agent_mcp_server.py"],
          "env": {
            "DASHSCOPE_API_KEY": "${DASHSCOPE_API_KEY}"
          }
        }
      }
    }

HTTP/SSE 模式（可选，供远程 MCP 客户端）：
    set TRAVEL_MCP_TRANSPORT=sse
    python scripts/travel_agent_mcp_server.py
    # 默认 http://127.0.0.1:8766/sse

环境变量：
    DASHSCOPE_API_KEY          — 必须（百炼 LLM）
    TRAVEL_MCP_TRANSPORT       — stdio（默认）| sse
    TRAVEL_MCP_HOST            — SSE 监听地址，默认 127.0.0.1
    TRAVEL_MCP_PORT            — SSE 端口，默认 8766
    TRAVEL_MCP_ENABLE_MEMORY   — 1/0，是否启用长期记忆，默认 1
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_CHAPTER8_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTER8_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER8_ROOT))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from mcp.server.fastmcp import FastMCP

from agent_framework.config import load_project_dotenv
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from agent_framework.tracing import setup_observability

load_project_dotenv()
setup_observability()

_HOST = os.getenv("TRAVEL_MCP_HOST", "127.0.0.1")
_PORT = int(os.getenv("TRAVEL_MCP_PORT", "8766"))
_ENABLE_MEMORY = os.getenv("TRAVEL_MCP_ENABLE_MEMORY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

mcp = FastMCP("travel-multi-agent", host=_HOST, port=_PORT)

_orchestrator: Optional[LangGraphOrchestrator] = None


def _get_orchestrator() -> LangGraphOrchestrator:
    """惰性初始化编排器（MCP 进程长驻，只创建一次）。"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator(
            enable_memory=_ENABLE_MEMORY,
            enable_guess_agent=True,
        )
    return _orchestrator


def _new_thread_id(prefix: str = "mcp") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@mcp.tool()
async def ask_travel_agent(query: str, thread_id: Optional[str] = None) -> str:
    """用 Chapter-8 多智能体回答旅行相关问题。

    能力：天气查询、酒店/餐厅推荐、航班、多日行程规划等。
    query: 用户的自然语言问题。
    thread_id: 可选会话 ID；同一 thread_id 可复用记忆上下文。
    """
    orchestrator = _get_orchestrator()
    tid = thread_id or _new_thread_id()
    result = await orchestrator.process_request(query, thread_id=tid)
    return (result.get("final_response") or "").strip() or json.dumps(
        result, ensure_ascii=False, default=str
    )


@mcp.tool()
async def ask_travel_agent_detailed(
    query: str,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """与 ask_travel_agent 相同，但返回结构化结果（含子任务、trace_id）。"""
    orchestrator = _get_orchestrator()
    tid = thread_id or _new_thread_id()
    result = await orchestrator.process_request(query, thread_id=tid)
    return {
        "thread_id": tid,
        "final_response": result.get("final_response", ""),
        "execution_plan": result.get("execution_plan"),
        "subtask_results": result.get("subtask_results"),
        "trace_id": result.get("trace_id"),
        "span_id": result.get("span_id"),
    }


def main() -> None:
    transport = os.getenv("TRAVEL_MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "sse":
        sse_url = f"http://{_HOST}:{_PORT}/sse"
        print(f"[travel-MCP] SSE 服务启动: {sse_url}", flush=True)
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
