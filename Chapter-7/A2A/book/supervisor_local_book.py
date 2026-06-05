"""
第 7 章示例：LangGraph Supervisor 调度远程 A2A 智能体

架构（三层）：
  1. Supervisor（本文件）— LLM 决策，通过 handoff 工具把任务交给子智能体
  2. A2A 子图（a2a_agents.py）— 每个远程服务包装成「单节点 LangGraph」
  3. 远端 A2A 服务（hotel_recommendation_agent/server.py）— 真正执行业务逻辑

运行前请先启动酒店 A2A 服务（终端 1）：
    cd Chapter-7/A2A/hotel_recommendation_agent
    python server.py --host 127.0.0.1 --port 9012

再运行 Supervisor（终端 2）：
    cd Chapter-7/A2A/book
    python supervisor_local_book.py

扩展更多 A2A 子智能体：编辑 a2a_agents.py 中的 A2A_AGENT_SPECS。
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

# ---------------------------------------------------------------------------
# 路径与导入：本书示例放在 book/ 子目录，需把上级 A2A 目录加入 sys.path
# ---------------------------------------------------------------------------
_A2A_DIR = Path(__file__).resolve().parent.parent  # Chapter-7/A2A
_BOOK_ROOT = _A2A_DIR.parent.parent.parent
if str(_A2A_DIR) not in sys.path:
    sys.path.insert(0, str(_A2A_DIR))

from a2a_agents import build_all_a2a_agent_graphs

# 加载 LLM / 百度地图等环境变量
load_dotenv(_A2A_DIR / "hotel_recommendation_agent" / ".env", override=False)
load_dotenv(_BOOK_ROOT / ".env", override=False)

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Supervisor 系统提示词
# {agent_list} 会在运行时填入 A2A_AGENT_SPECS 里的子智能体列表
# ---------------------------------------------------------------------------
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
    """
    为每个 A2A 子智能体生成一个 handoff 工具。

    Supervisor LLM 调用 transfer_to_hotel_agent 等工具时，
    LangGraph 会自动跳转到同名子图（见 a2a_agents.build_a2a_agent_graph）。
    """
    return [
        create_handoff_tool(
            agent_name=node,  # 必须与 a2a_agents 里子图节点名一致
            description=f"交给远程 A2A {desc}（{url}）",
        )
        for node, url, desc in A2A_AGENT_SPECS
    ]


def _create_llm() -> ChatOpenAI:
    """创建 Supervisor 使用的 LLM（默认通义千问兼容 OpenAI 接口）。"""
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
    组装并编译 Supervisor 图。

    调度拓扑示意：

        用户输入
            │
            ▼
    ┌─────────────────────────┐
    │    supervisor           │  ← LLM 读 messages，决定 handoff 或直接回复
    │  (LLM 决策中心)         │
    └─────────┬───────────────┘
              │
        ┌─────┼────────────────────────────┐
        │             │                    │
    handoff         handoff               直接回复
    hotel_agent    weather_agent             │
        │              │                     ▼
        ▼              ▼                 ┌───────────┐
    ┌──────────┐  ┌─────────────┐        │    END    │
    │hotel_agent│  │weather_agent│        └───────────┘
    │ 子图(1节点)│  │ 子图(1节点) │
    └─────┬─────┘  └──────┬──────┘
          │               │
          └───────┬───────┘
                  ▼
          子图返回 AIMessage，回到 supervisor
    """
    # 把注册表渲染进 Prompt，让 LLM 知道有哪些子智能体可用
    agent_list = "\n".join(f"- {node}: {desc}（{url}）" for node, url, desc in A2A_AGENT_SPECS)

    # 每个 A2A URL → 一个单节点 LangGraph 子图
    sub_graphs = build_all_a2a_agent_graphs()

    # forward 工具：Supervisor 需要时可把用户原话转发给子智能体
    forward = create_forward_message_tool(supervisor_name="supervisor")

    supervisor = create_supervisor(
        agents=sub_graphs,                          # 可 handoff 的子图列表
        model=llm,
        tools=[forward] + _build_handoff_tools(),   # LLM 可调用的工具
        prompt=SUPERVISOR_PROMPT.format(agent_list=agent_list),
        supervisor_name="supervisor",
        output_mode="full_history",                 # 保留完整对话，便于调试调度链
    )

    # MemorySaver：同一 thread_id 下多轮对话共享 checkpoint
    return supervisor.compile(checkpointer=MemorySaver())


def _pick_final_response(messages: List[Any]) -> str:
    """
    从 messages 里挑出最终展示给用户的文本。

    优先级：Supervisor 最后一次纯文本回复 > 子智能体最后一次回复。
    跳过 tool_calls 消息和 "Transferring..." 等调度话术。
    """
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
    """打印本次请求经过了哪些 handoff（便于学生观察调度路径）。"""
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
    """
    流式监听图执行事件，打印阶段进度。

    A2A 调用通常需 30s~2min，进度日志可避免学生误以为程序卡死。
    """
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
            # handoff 工具被调用，即将进入对应 A2A 子图
            tool = name or event.get("data", {}).get("input", "")
            last_hint = f"handoff 工具: {tool or name}"
            print(f"  … {last_hint} ({time.perf_counter() - t0:.0f}s)", flush=True)
        elif kind == "on_chain_start" and name in {n for n, _, _ in A2A_AGENT_SPECS}:
            last_hint = f"子图 {name} 启动（内部会 HTTP 调用远端 A2A）"
            print(f"  … {last_hint} ({time.perf_counter() - t0:.0f}s)", flush=True)
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            result = event.get("data", {}).get("output") or {}

    if result is None:
        raise RuntimeError(f"Supervisor 未正常结束（最后阶段: {last_hint or 'unknown'}）")
    print(f"  ✓ 调度完成 ({time.perf_counter() - t0:.1f}s)", flush=True)
    return result


async def run_interactive(app: Any, thread_id: str = "a2a_supervisor") -> None:
    """命令行交互：读用户输入 → 调度 → 打印结果。"""
    print("=== LangGraph Supervisor + A2A 远程智能体 ===")
    print("输入 'exit' 退出对话\n")
    for node, url, desc in A2A_AGENT_SPECS:
        print(f"  - {node}: {desc} @ {url}")
    print()

    # thread_id 用于 checkpoint 与 A2A context_id 关联（见 a2a_agents._a2a_context_store）
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
