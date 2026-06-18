"""将 A2A 远程 Agent 包装为 LangGraph 子图，供 Supervisor handoff 调度。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.transport.a2a.call_traced import a2a_call_remote

# thread_id -> node_name -> a2a context_id
_a2a_context_store: Dict[str, Dict[str, str]] = {}


def _extract_query(messages: List[Any]) -> str:
    parts: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            parts.append(str(msg.content))
            if len(parts) >= 2:
                break
    return "\n".join(reversed(parts)) if parts else ""


def build_a2a_agent_graph(endpoint: A2AEndpoint) -> Any:
    """单个 A2A 远程 Agent → 单节点 LangGraph。"""
    node_name = endpoint.node_name
    url = endpoint.url.strip()
    description = endpoint.description or f"A2A {node_name}"

    async def run_a2a_agent(state: MessagesState, config) -> Dict[str, Any]:
        query = _extract_query(state["messages"])
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        ctx_bucket = _a2a_context_store.setdefault(thread_id, {})
        context_id: Optional[str] = ctx_bucket.get(node_name)

        content, new_context_id, _success = await a2a_call_remote(
            url,
            query or "请根据上下文完成任务",
            context_id,
        )
        if new_context_id:
            ctx_bucket[node_name] = new_context_id

        return {
            "messages": [
                AIMessage(
                    content=content,
                    name=node_name,
                    additional_kwargs={
                        "a2a_endpoint": url,
                        "description": description,
                        "transport": "a2a",
                    },
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node(node_name, run_a2a_agent)
    graph.set_entry_point(node_name)
    graph.add_edge(node_name, END)
    return graph.compile(name=node_name)


def build_a2a_agent_graphs(endpoints: List[A2AEndpoint]) -> List[Any]:
    return [build_a2a_agent_graph(ep) for ep in endpoints if ep.is_configured()]
