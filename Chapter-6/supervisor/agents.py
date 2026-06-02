"""将 Chapter-6 本地子智能体包装为 Supervisor 可调度的 LangGraph 节点"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

SUP_DIR = Path(__file__).resolve().parent
if str(SUP_DIR) not in sys.path:
    sys.path.insert(0, str(SUP_DIR))

from _ch6_loader import import_pip_langgraph
from sub_agents import SubAgentFactory

_lg_graph = import_pip_langgraph("graph")
StateGraph = _lg_graph.StateGraph
MessagesState = _lg_graph.MessagesState
END = _lg_graph.END

# supervisor handoff 使用的 agent 名称（snake_case，与 create_handoff_tool 一致）
AGENT_SPECS = [
    ("weather_agent", "WeatherAgent", "天气查询"),
    ("attraction_agent", "AttractionAgent", "景点推荐"),
    ("hotel_agent", "HotelAgent", "酒店推荐"),
    ("restaurant_agent", "RestaurantAgent", "美食推荐"),
    ("flight_agent", "FlightAgent", "航班查询"),
    ("itinerary_agent", "ItineraryAgent", "行程规划"),
]


def _extract_query(messages: List[Any]) -> str:
    """从 supervisor 转交的消息中提取子任务指令"""
    parts: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            parts.append(str(msg.content))
            if len(parts) >= 2:
                break
    return "\n".join(reversed(parts)) if parts else ""


def _parse_agent_result(state: Dict[str, Any]) -> str:
    tool_outputs = []
    agent_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type"):
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


def build_sub_agent_graph(node_name: str, factory_name: str, description: str):
    """为每个 SubAgent 构建单节点 StateGraph，供 create_supervisor 调度"""

    async def run_agent(state: MessagesState) -> Dict[str, Any]:
        query = _extract_query(state["messages"])
        agent = SubAgentFactory.get_agent(factory_name)
        thread_id = "supervisor_sub"
        result = await agent.ainvoke(
            {"messages": [("user", query or "请根据上下文完成任务")]},
            {"configurable": {"thread_id": f"{thread_id}_{node_name}"}},
        )
        content = _parse_agent_result(result)
        return {
            "messages": [
                AIMessage(
                    content=content,
                    name=node_name,
                    additional_kwargs={"agent": factory_name, "description": description},
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node(node_name, run_agent)
    graph.set_entry_point(node_name)
    graph.add_edge(node_name, END)
    return graph.compile(name=node_name)


def build_all_sub_agent_graphs() -> List[Any]:
    graphs = []
    for node_name, factory_name, desc in AGENT_SPECS:
        graphs.append(build_sub_agent_graph(node_name, factory_name, desc))
    return graphs
