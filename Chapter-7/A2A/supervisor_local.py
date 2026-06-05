"""
LangGraph Supervisor + 远程 A2A 智能体调度。

- 使用 langgraph_supervisor.create_supervisor 作为调度器
- 子智能体为 HTTP A2A 服务（默认 hotel @ 9012），可在 a2a_agents.A2A_AGENT_SPECS 扩展

运行:
    cd Chapter-7/A2A
    python supervisor_local_book.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor.handoff import create_forward_message_tool, create_handoff_tool
from langgraph_supervisor.supervisor import create_supervisor
from typing import Any, List, Tuple
from a2a_agents import A2A_AGENT_SPECS, build_all_a2a_agent_graphs

_A2A_DIR = Path(__file__).resolve().parent
_BOOK_ROOT = _A2A_DIR.parent.parent.parent
load_dotenv(_A2A_DIR / "hotel_recommendation_agent" / ".env", override=False)
load_dotenv(_BOOK_ROOT / ".env", override=False)

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SUPERVISOR_PROMPT = """你是 A2A 多智能体调度 Supervisor，负责把用户请求分派给远程专业智能体并整合结果。

## 可用子智能体（handoff 名称必须完全一致）
{agent_list}

## 规则
1. 严格匹配用户请求：只问酒店就只调 hotel_agent，不要擅自扩展无关任务
2. 每次 handoff 给出完整独立指令（城市、日期、预算、偏好等）
3. 可依次调用多个子智能体，但仅当用户明确需要多项信息时
4. 子智能体返回后：完整呈现具体结果（名称、地址、价格、评分、理由），禁止空泛概述
5. 禁止输出调度话术（Transferring / handoff 等）
6. 使用中文，语气友好专业
"""

A2A_AGENT_SPECS: List[Tuple[str, str, str]] = [
    ("hotel_agent", "http://127.0.0.1:9012/", "酒店推荐（A2A）"),
    # ("weather_agent", "http://127.0.0.1:9013/", "天气查询（A2A）"),
]

def _build_handoff_tools() -> List[Any]:
    return [
        create_handoff_tool(
            agent_name=node,
            description=f"交给远程 A2A {desc}（{url}）",
        )
        for node, url, desc in A2A_AGENT_SPECS
    ]


def _create_llm() -> ChatOpenAI:
    api_key = (
        os.getenv("CHAT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("请设置 CHAT_API_KEY 或 DASHSCOPE_API_KEY")

    base_url = (
        os.getenv("CHAT_ENDPOINT")
        or os.getenv("DASHSCOPE_CHAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = os.getenv("DEPLOYMENT_NAME") or os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen-plus"
    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in ("0", "false", "no")

    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        request_timeout=90,
        max_retries=2,
        http_client=httpx.Client(verify=ssl_verify),
        http_async_client=httpx.AsyncClient(verify=ssl_verify),
    )


def _build_supervisor_app(llm: ChatOpenAI) -> Any:
    """
        用户输入
        │
        ▼
┌─────────────────────────┐
│    supervisor           │
│  (LLM 决策中心)         │
└─────────┬───────────────┘
          │
    ┌─────┼────────────────────────────┐
    │             │                    │
handoff         handoff               直接回复
hotel_agent    weather_agent             │
    │              │                     │
    ▼              ▼                     ▼
┌─────────--┐  ┌─────────────┐    ┌───────────┐
│hotel_agent│  │weather_agent│    │    END    │
└─────┬─────┘  └──────┬──────┘    └───────────┘
      │               │
      └───────┬───────┘
              │
              ▼
      回到 supervisor
    """
    agent_list = "\n".join(f"- {node}: {desc}（{url}）" for node, url, desc in A2A_AGENT_SPECS)
    sub_graphs = build_all_a2a_agent_graphs()
    forward = create_forward_message_tool(supervisor_name="supervisor")
    supervisor = create_supervisor(
        agents=sub_graphs,
        model=llm,
        tools=[forward] + _build_handoff_tools(),
        prompt=SUPERVISOR_PROMPT.format(agent_list=agent_list),
        supervisor_name="supervisor",
        output_mode="full_history",
    )
    return supervisor.compile(checkpointer=MemorySaver())


def _pick_final_response(messages: List[Any]) -> str:
    agent_names = {n for n, _, _ in A2A_AGENT_SPECS}
    last_agent: Optional[str] = None
    last_supervisor: Optional[str] = None

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        if getattr(msg, "tool_calls", None):
            continue
        name = getattr(msg, "name", None)
        text = str(msg.content).strip()
        if not text or text.lower().startswith("transferring"):
            continue
        if name in agent_names and last_agent is None:
            last_agent = text
        elif name in (None, "supervisor") and last_supervisor is None:
            last_supervisor = text

    return last_supervisor or last_agent or ""


def _print_routing(messages: List[Any]) -> None:
    agent_names = {n for n, _, _ in A2A_AGENT_SPECS}
    chain: List[str] = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if isinstance(msg, AIMessage) and name in agent_names:
            chain.append(f"supervisor → {name}")
    if chain:
        print("\n【调度链】")
        for step in chain:
            print(f"  {step}")
    else:
        print("\n【调度链】 Supervisor 直接回复（未 handoff）")


async def _invoke_with_progress(app: Any, user_input: str, config: dict) -> dict:
    """ainvoke + 阶段进度，避免长时间无输出像卡死。"""
    import time

    t0 = time.perf_counter()
    result: dict | None = None
    last_hint = ""

    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        version="v2",
    ):
        kind = event.get("event", "")
        name = event.get("name") or ""

        if kind == "on_chat_model_start" and "supervisor" in name.lower():
            last_hint = "Supervisor LLM 决策中"
            print(f"  … {last_hint} ({time.perf_counter() - t0:.0f}s)", flush=True)
        elif kind == "on_tool_start":
            tool = name or event.get("data", {}).get("input", "")
            last_hint = f"handoff 工具: {tool or name}"
            print(f"  … {last_hint} ({time.perf_counter() - t0:.0f}s)", flush=True)
        elif kind == "on_chain_start" and name in {n for n, _, _ in A2A_AGENT_SPECS}:
            last_hint = f"子图 {name} 启动"
            print(f"  … {last_hint} ({time.perf_counter() - t0:.0f}s)", flush=True)
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            result = event.get("data", {}).get("output") or {}

    if result is None:
        raise RuntimeError(f"Supervisor 未正常结束（最后阶段: {last_hint or 'unknown'}）")
    print(f"  ✓ 调度完成 ({time.perf_counter() - t0:.1f}s)", flush=True)
    return result


async def run_interactive(app: Any, thread_id: str = "a2a_supervisor") -> None:
    print("=== LangGraph Supervisor + A2A 远程智能体 ===")
    print("输入 'exit' 退出对话\n")
    for node, url, desc in A2A_AGENT_SPECS:
        print(f"  - {node}: {desc} @ {url}")
    print()

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        print("\n⏳ Supervisor 调度中（含 LLM + 远程 A2A，通常 1~2 分钟）...", flush=True)
        try:
            result = await _invoke_with_progress(app, user_input, config)
            messages = result.get("messages", [])
            _print_routing(messages)
            final = _pick_final_response(messages)
            print(f"\nA2A: {final or '（无回复）'}\n")
        except Exception as exc:
            print(f"\n❌ 出错: {exc}\n")


async def main() -> None:
    llm = _create_llm()
    app = _build_supervisor_app(llm)
    await run_interactive(app)


if __name__ == "__main__":
    asyncio.run(main())
