"""LangGraph Supervisor 图：动态 handoff + MemorySaver 短期记忆"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor.handoff import create_handoff_tool
from langgraph_supervisor.supervisor import create_supervisor

SUP_DIR = Path(__file__).resolve().parent
if str(SUP_DIR) not in sys.path:
    sys.path.insert(0, str(SUP_DIR))

import bootstrap  # noqa: E402

bootstrap.setup()

from agents import AGENT_SPECS, build_all_sub_agent_graphs

SUPERVISOR_PROMPT = """你是旅行多智能体调度 Supervisor，负责把用户请求分派给专业子智能体并整合结果。

## 可用子智能体（通过 handoff 工具调用）
- weather_agent：天气查询
- attraction_agent：景点推荐
- hotel_agent：酒店推荐
- restaurant_agent：美食推荐
- flight_agent：航班查询
- itinerary_agent：行程规划

## 规则
1. 严格匹配用户请求范围：只问天气就只调 weather_agent，不要擅自扩展成完整旅行规划
2. 每次 handoff 时，给子智能体一条完整、独立的指令（含地点、日期、预算等已知信息）
3. 可依次调用多个子智能体，但仅当用户明确需要多项信息时
4. 子智能体返回后，若已满足用户请求，直接输出最终答案并停止
5. 禁止输出调度过程话术（如 "Transferring to..."）
6. 使用中文，语气友好专业
"""


def _build_handoff_tools() -> List[Any]:
    tools = []
    for node_name, _factory, desc in AGENT_SPECS:
        tools.append(
            create_handoff_tool(
                agent_name=node_name,
                description=f"交给 {desc} 子智能体处理",
            )
        )
    return tools


def build_supervisor_app(
    llm: ChatOpenAI,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Any:
    """编译 Supervisor 多智能体应用（含 MemorySaver 短期记忆，可选 Store 长期记忆）"""
    sub_graphs = build_all_sub_agent_graphs()
    supervisor = create_supervisor(
        agents=sub_graphs,
        model=llm,
        tools=_build_handoff_tools(),
        prompt=SUPERVISOR_PROMPT,
        supervisor_name="supervisor",
        output_mode="full_history",
    )
    if checkpointer is None:
        checkpointer = MemorySaver()
    compile_kwargs: Dict[str, Any] = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return supervisor.compile(**compile_kwargs)
