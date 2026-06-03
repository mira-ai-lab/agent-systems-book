"""
本地 LangGraph Supervisor — 单文件交互入口

- 单任务：Supervisor 动态 handoff（演示 Supervisor 模式）
- 路由：先 TaskPlanner build_plan，子任务 >1 走规划流水线，=1 走 Supervisor
- 依赖同目录: sub_agents.py, planned_pipeline.py, travel_common.py, .env

运行:
    cd supervisor
    python local_supervisor.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 路径 & pip langgraph_demo（防止本地 langgraph_demo/ 目录遮蔽 site-packages）
# ---------------------------------------------------------------------------
SUP_DIR = Path(__file__).resolve().parent
BOOK_ROOT = SUP_DIR.parent

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

if str(SUP_DIR) not in sys.path:
    sys.path.insert(0, str(SUP_DIR))


def _import_pip_langgraph(submodule: str = ""):
    saved = sys.path[:]
    blocked = {str(SUP_DIR), str(BOOK_ROOT / "langgraph_demo")}
    sys.path[:] = [p for p in sys.path if p not in blocked]
    try:
        name = f"langgraph_demo.{submodule}" if submodule else "langgraph_demo"
        return importlib.import_module(name)
    finally:
        sys.path[:] = saved


_import_pip_langgraph()
_import_pip_langgraph("_internal")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph_supervisor.handoff import create_forward_message_tool, create_handoff_tool
from langgraph_supervisor.supervisor import create_supervisor

_lg_graph = _import_pip_langgraph("graph")
_lg_ckpt = _import_pip_langgraph("checkpoint.memory")
StateGraph = _lg_graph.StateGraph
MessagesState = _lg_graph.MessagesState
END = _lg_graph.END
MemorySaver = _lg_ckpt.MemorySaver

load_dotenv(SUP_DIR / ".env")
load_dotenv(BOOK_ROOT / ".env")

from sub_agents import SubAgentFactory  # noqa: E402
from planned_pipeline import PlannedPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# 本地子智能体规格 & 图构建
# ---------------------------------------------------------------------------
AGENT_SPECS: List[tuple[str, str, str]] = [
    ("weather_agent", "WeatherAgent", "天气查询"),
    ("attraction_agent", "AttractionAgent", "景点推荐"),
    ("hotel_agent", "HotelAgent", "酒店推荐"),
    ("restaurant_agent", "RestaurantAgent", "美食推荐"),
    ("flight_agent", "FlightAgent", "航班查询"),
    ("itinerary_agent", "ItineraryAgent", "行程规划"),
]

SUPERVISOR_PROMPT = """你是旅行多智能体 Supervisor，负责调度本地子智能体并整合结果。

## 可用子智能体（handoff 名称必须完全一致）
{agent_list}

## 规则
1. 严格匹配用户请求：只问天气就只调 weather_agent，不要擅自扩展成完整旅行规划
2. 每次 handoff 给出完整独立指令（地点、日期、预算、偏好等）
3. 可依次调用多个子智能体，但仅当用户明确需要多项信息时
4. 子智能体返回后：必须**完整呈现**其具体结果（如酒店/景点/餐厅的名称、地址、价格、评分、推荐理由），禁止用「已为您精选 N 家，信息含地址价格」等空泛概述代替条目列表；可在末尾加 1 句简短结语
5. 禁止输出调度话术（Transferring / handoff 等）
6. 使用中文，语气友好专业

注意：多步骤旅行规划（天气+机票+景点+行程等）由系统自动走规划流水线，不会进入本 Supervisor 路径。
"""


def _parse_agent_result(state: Dict[str, Any]) -> str:
    tool_outputs: List[Any] = []
    agent_text = ""
    for msg in state.get("messages", []):
        if not hasattr(msg, "type"):
            continue
        if msg.type == "tool" and getattr(msg, "content", None):
            try:
                tool_outputs.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_outputs.append(msg.content)
        elif msg.type == "ai" and getattr(msg, "content", None):
            agent_text = msg.content
    if agent_text.strip():
        return agent_text.strip()
    if tool_outputs:
        return json.dumps(tool_outputs[-1], ensure_ascii=False, indent=2)
    return "（子智能体未返回有效内容）"


def _extract_user_query(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return str(msg.content).strip()
    return ""


def _agent_user_message(query: str) -> str:
    if not query:
        return "请根据上下文完成任务"
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"{query}\n\n"
        f"[系统参考：当前日期 {today}。"
        f"涉及「今天/明天/后天」时请按此日期换算，"
        f"get_weather 的 date 可直接传「今天」等相对词]"
    )


def _build_sub_agent_graph(node_name: str, factory_name: str, description: str) -> Any:
    async def run_agent(state: MessagesState) -> Dict[str, Any]:
        query = _extract_user_query(state["messages"])
        print(f"\n  ▶ [{node_name}] 执行 {factory_name}...", flush=True)
        agent = SubAgentFactory.get_agent(factory_name)
        result = await agent.ainvoke(
            {"messages": [("user", _agent_user_message(query))]},
            {"configurable": {"thread_id": f"local_sup_{node_name}_{uuid.uuid4().hex[:8]}"}},
        )
        content = _parse_agent_result(result)
        print(f"  ✓ [{node_name}] 完成", flush=True)
        return {
            "messages": [
                AIMessage(
                    content=content,
                    name=node_name,
                    additional_kwargs={"agent": factory_name, "description": description},
                )
            ]
        }

    g = StateGraph(MessagesState)
    g.add_node(node_name, run_agent)
    g.set_entry_point(node_name)
    g.add_edge(node_name, END)
    return g.compile(name=node_name)


def _build_handoff_tools() -> List[Any]:
    return [
        create_handoff_tool(
            agent_name=node,
            description=f"交给 {desc} 子智能体（本地 sub_agents.py / {factory}）",
        )
        for node, factory, desc in AGENT_SPECS
    ]


def _create_llm() -> ChatOpenAI:
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("STRANDS_OPENAI_API_KEY")
    api_key = dashscope_key or openai_key
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    if os.getenv("DASHSCOPE_CHAT_BASE_URL"):
        base_url = os.getenv("DASHSCOPE_CHAT_BASE_URL", "").rstrip("/")
    elif dashscope_key and api_key == dashscope_key:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    else:
        base_url = (
            os.getenv("STRANDS_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")

    if os.getenv("DASHSCOPE_CHAT_MODEL"):
        model = os.getenv("DASHSCOPE_CHAT_MODEL", "")
    elif dashscope_key and api_key == dashscope_key:
        model = "qwen-plus"
    else:
        model = (
            os.getenv("LANGGRAPH_MODEL_ID")
            or os.getenv("STRANDS_MODEL_ID")
            or os.getenv("OPENAI_MODEL")
            or "qwen-plus"
        )
    import httpx

    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in (
        "0",
        "false",
        "no",
    )
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=ssl_verify),
        http_async_client=httpx.AsyncClient(verify=ssl_verify),
    )


def _build_supervisor_app(llm: ChatOpenAI) -> Any:
    agent_list = "\n".join(
        f"- {node}: {desc}（{factory}）" for node, factory, desc in AGENT_SPECS
    )
    sub_graphs = [_build_sub_agent_graph(n, f, d) for n, f, d in AGENT_SPECS]
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


def _has_itemized_list(text: str) -> bool:
    if re.search(r"^\s*[1-9]\.", text, re.MULTILINE):
        return True
    if text.count("**") >= 4:
        return True
    return False


def _looks_like_meta_only(text: str) -> bool:
    """声称推荐了多条，但没有具体列表。"""
    if _has_itemized_list(text):
        return False
    return bool(re.search(r"已为您(?:精选|推荐|筛选).*?\d+家", text))


def _pick_final_response(messages: List[Any]) -> str:
    agent_names = {n for n, _, _ in AGENT_SPECS}
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
        if last_agent is not None and last_supervisor is not None:
            break

    if last_supervisor and _looks_like_meta_only(last_supervisor) and last_agent:
        return last_agent
    if last_supervisor and last_agent:
        if _has_itemized_list(last_agent) and not _has_itemized_list(last_supervisor):
            return last_agent
        if len(last_agent) > int(len(last_supervisor) * 1.2):
            return last_agent
    return last_supervisor or last_agent or ""


def _last_ai_text(messages: List[Any]) -> str:
    return _pick_final_response(messages)


def _print_routing(messages: List[Any]) -> None:
    agent_names = {n for n, _, _ in AGENT_SPECS}
    chain: List[str] = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if isinstance(msg, AIMessage) and name in agent_names:
            chain.append(f"supervisor → {name}")
    if chain:
        print("\n【调度链】", flush=True)
        for step in chain:
            print(f"  {step}", flush=True)
    else:
        print("\n【调度链】 Supervisor 直接回复（未 handoff 子智能体）", flush=True)


DEFAULT_RECURSION_LIMIT = 50


async def _dispatch_query(
    app: Any,
    pipeline: PlannedPipeline,
    query: str,
    thread_id: str,
) -> tuple[str, str]:
    """
    用 build_plan 子任务数路由，返回 (mode, final_text)。
    mode: planned | supervisor
    """
    route, plan_ctx = await pipeline.classify_route(query)
    if route == "planned":
        print("🔀 多子任务 → 规划流水线（execute_layer + aggregate）", flush=True)
        final = await pipeline.run(
            query,
            thread_id=f"planned_{thread_id}",
            plan_ctx=plan_ctx,
        )
        print(
            "\n【调度链】 规划模式（TaskPlanner 分层执行，无 Supervisor handoff）",
            flush=True,
        )
        return "planned", final

    print("🔀 单子任务 → Supervisor 动态 handoff", flush=True)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    messages = result.get("messages", [])
    _print_routing(messages)
    return "supervisor", _last_ai_text(messages)


# ---------------------------------------------------------------------------
# 交互主循环
# ---------------------------------------------------------------------------
async def run_interactive(
    app: Any,
    pipeline: PlannedPipeline,
    thread_id: str = "interactive",
) -> None:
    print("=" * 60)
    print("本地 LangGraph Supervisor（6 个子智能体）")
    print("输入问题后回车；输入 quit / exit / q 退出")
    print("=" * 60)
    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q", "退出"):
            print("再见。")
            break

        print("\n⏳ 处理中...", flush=True)
        try:
            _, final = await _dispatch_query(app, pipeline, query, thread_id)
            print("\n" + "-" * 60)
            print("Assistant:")
            print(final or "（无回复）")
            print("-" * 60)
        except Exception as exc:
            print(f"\n❌ 出错: {exc}", flush=True)


async def run_once(
    app: Any,
    query: str,
    thread_id: str = "single",
    llm: Optional[ChatOpenAI] = None,
    pipeline: Optional[PlannedPipeline] = None,
) -> str:
    pipe = pipeline or PlannedPipeline(llm or _create_llm())
    _, final = await _dispatch_query(app, pipe, query, thread_id)
    return final


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="本地 Supervisor 交互式多智能体")
    parser.add_argument("-q", "--query", help="单条问题（非交互模式）")
    parser.add_argument(
        "--thread-id",
        default="local_supervisor",
        help="会话 thread_id（MemorySaver 短期记忆）",
    )
    args = parser.parse_args()

    print("初始化 LLM、Supervisor 与规划流水线...", flush=True)
    llm = _create_llm()
    app = _build_supervisor_app(llm)
    pipeline = PlannedPipeline(llm)
    print("✓ 就绪\n", flush=True)

    if args.query:
        text = asyncio.run(run_once(app, args.query, args.thread_id, llm=llm, pipeline=pipeline))
        print("\n" + text)
    else:
        asyncio.run(run_interactive(app, pipeline, args.thread_id))


if __name__ == "__main__":
    main()
