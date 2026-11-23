"""Supervisor 子 Agent 调用（带 latc span）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.tracing.trace_provider import current_trace_add_event, span_name, trace_span


def _parse_agent_result(state: Dict[str, Any]) -> str:
    tool_outputs: List[Any] = []
    agent_text = ""
    for msg in state.get("messages", []):
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and getattr(msg, "content", None):
            try:
                tool_outputs.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_outputs.append(msg.content)
        elif msg_type == "ai" and getattr(msg, "content", None):
            agent_text = str(msg.content)
    if agent_text.strip():
        return agent_text.strip()
    if tool_outputs:
        return json.dumps(tool_outputs[-1], ensure_ascii=False, indent=2)
    return "（子智能体未返回有效内容）"


@trace_span(
    name=span_name("agent.invoke"),
    attrs_args=["node_name", "factory_name", "query"],
    record_result=False,
)
async def invoke_local_sub_agent(
    registry: SubAgentRegistry,
    *,
    factory_name: str,
    node_name: str,
    description: str,
    query: str,
) -> str:
    agent = registry.get_agent(factory_name)
    result = await agent.ainvoke(
        {"messages": [("user", query or "请根据上下文完成任务")]},
        {"configurable": {"thread_id": f"supervisor_{node_name}"}},
    )
    content = _parse_agent_result(result)
    current_trace_add_event(
        "sub_agent_conversation",
        {
            "transport": "local",
            "node_name": node_name,
            "agent": factory_name,
            "description": description[:120],
            "query": (query or "")[:500],
            "response": content[:500],
            "status": "completed",
        },
    )
    return content
