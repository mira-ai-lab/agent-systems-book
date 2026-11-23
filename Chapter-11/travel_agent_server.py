"""End-to-end travel agent server.

Publishes the Chapter-8 travel multi-agent system as callable tools for coding agents (Claude Code / Cursor / Codex); MCP is only the transport.
The orchestrator delegates to the five A2A
domain-agent services, which must already be running before the first tool call
(see Chapter-8/travel_multi_agent.ipynb or ``python -m travel_multi_agent.a2a.server``).

:author: zhanghong, liuhui, xingrui
:date: 2026-11-23
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_BOOK_ROOT = Path(__file__).resolve().parent.parent
_CHAPTER8_DIR = _BOOK_ROOT / "Chapter-8"
for _p in (_CHAPTER8_DIR, _BOOK_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mcp.server.fastmcp import FastMCP

# stdio MCP 用 stdout 传输协议消息，必须关掉 span 的 console 输出，
# 并把结构化日志压到 WARNING，避免污染 stdout。
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from travel_multi_agent.config import load_project_dotenv
from travel_multi_agent.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from travel_multi_agent.tracing import setup_observability

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
    """Lazily create the orchestrator once per long-lived MCP process."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator(enable_memory=_ENABLE_MEMORY)
    return _orchestrator


def _new_thread_id(prefix: str = "mcp") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@mcp.tool()
async def ask_travel_agent(query: str, thread_id: Optional[str] = None) -> str:
    """用旅行多智能体系统回答旅行相关问题（天气 / 酒店 / 餐厅 / 航班 / 行程规划）。

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
    """同 ask_travel_agent，但额外返回执行计划、子任务结果与 trace_id。"""
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
        print(f"[travel-MCP] SSE 服务启动: http://{_HOST}:{_PORT}/sse", flush=True)
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()