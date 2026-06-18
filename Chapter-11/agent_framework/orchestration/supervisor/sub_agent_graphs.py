"""将 SubAgentRegistry 中的子 Agent 包装为 Supervisor 可调度的 LangGraph 子图。"""



from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph



from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.supervisor.agent_names import registry_agent_to_node_name
from agent_framework.orchestration.supervisor.invoke_traced import invoke_local_sub_agent





def _extract_query(messages: List[Any]) -> str:
    from langchain_core.messages import HumanMessage
    parts: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            parts.append(str(msg.content))
            if len(parts) >= 2:
                break
    return "\n".join(reversed(parts)) if parts else ""





def build_sub_agent_graph(
    node_name: str,
    registry: SubAgentRegistry,
    factory_name: str,
    description: str,
) -> Any:

    async def run_agent(state: MessagesState) -> Dict[str, Any]:
        query = _extract_query(state["messages"])
        content = await invoke_local_sub_agent(
            registry,
           factory_name=factory_name,
            node_name=node_name,
            description=description,
            query=query,
        )
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





def build_sub_agent_graphs(registry: SubAgentRegistry) -> List[Any]:
    graphs: List[Any] = []
    for factory_name in registry.get_agent_names():
        if registry.is_metadata_only(factory_name):
            continue
        info = registry.agents.get(factory_name, {})
        node_name = registry_agent_to_node_name(factory_name)
        description = str(info.get("description") or factory_name)
        graphs.append(
            build_sub_agent_graph(node_name, registry, factory_name, description)
        )
    return graphs

