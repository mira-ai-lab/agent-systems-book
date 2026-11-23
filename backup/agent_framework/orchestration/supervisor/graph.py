"""Supervisor 图编译（langgraph_supervisor + handoff + 可选 A2A）。"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from langchain_openai import ChatOpenAI

from agent_framework.domain.a2a_spec import A2AEndpoint
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.protocol import TRANSPORT_A2A, TRANSPORT_LOCAL, TRANSPORT_MIXED, AgentTransport
from agent_framework.orchestration.supervisor.agent_names import registry_agent_to_node_name
from agent_framework.orchestration.supervisor.sub_agent_graphs import build_sub_agent_graphs
from agent_framework.transport.a2a.agent_graphs import build_a2a_agent_graphs


def _require_supervisor_deps() -> None:
    try:
        import langgraph_supervisor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Supervisor 模式需要安装可选依赖：pip install 'agent-platform[supervisor]'"
        ) from exc


def _active_a2a_endpoints(endpoints: Sequence[A2AEndpoint]) -> List[A2AEndpoint]:
    return [ep for ep in endpoints if ep.is_configured()]


def _registry_agents_replaced_by_a2a(
    registry: SubAgentRegistry,
    endpoints: Sequence[A2AEndpoint],
) -> set[str]:
    configured = _active_a2a_endpoints(endpoints)
    return {
        ep.registry_agent
        for ep in configured
        if ep.registry_agent and ep.registry_agent in registry.agents
    }


def resolve_supervisor_subgraphs(
    registry: SubAgentRegistry,
    *,
    transport: AgentTransport = TRANSPORT_LOCAL,
    a2a_endpoints: Sequence[A2AEndpoint] = (),
) -> Tuple[List[Any], List[Tuple[str, str]]]:
    """返回 (子图列表, handoff 元数据[(node_name, description)])。"""
    active_a2a = _active_a2a_endpoints(a2a_endpoints)
    replaced = _registry_agents_replaced_by_a2a(registry, active_a2a) if transport == TRANSPORT_MIXED else set()

    graphs: List[Any] = []
    handoff_meta: List[Tuple[str, str]] = []

    if transport in (TRANSPORT_LOCAL, TRANSPORT_MIXED):
        if transport == TRANSPORT_LOCAL:
            local_registry = registry
        else:
            local_registry = registry.subset_excluding(replaced)
        if local_registry.get_agent_names():
            graphs.extend(build_sub_agent_graphs(local_registry))
            for factory_name in local_registry.get_agent_names():
                if local_registry.is_metadata_only(factory_name):
                    continue
                node = registry_agent_to_node_name(factory_name)
                desc = local_registry.agents.get(factory_name, {}).get("description", factory_name)
                handoff_meta.append((node, str(desc)))

    if transport in (TRANSPORT_A2A, TRANSPORT_MIXED):
        seen_nodes: set[str] = {n for n, _ in handoff_meta}
        for ep in active_a2a:
            if transport == TRANSPORT_MIXED and ep.registry_agent in replaced:
                pass  # already skipped local
            graphs.extend(build_a2a_agent_graphs([ep]))
            if ep.node_name not in seen_nodes:
                label = ep.description or f"A2A {ep.node_name}"
                handoff_meta.append((ep.node_name, f"{label}（远程 A2A）"))
                seen_nodes.add(ep.node_name)

    if transport == TRANSPORT_A2A and not active_a2a:
        raise ValueError("transport='a2a' 需要领域插件配置至少一个有效 A2AEndpoint（url 非空）")

    if not graphs:
        raise ValueError("Supervisor 无可用子 Agent：请检查 registry 与 A2A 端点配置")

    return graphs, handoff_meta


def build_default_supervisor_prompt(
    registry: SubAgentRegistry,
    *,
    handoff_meta: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    lines = [
        "你是多智能体调度 Supervisor，负责把用户请求分派给专业子智能体并整合结果。",
        "",
        "## 可用子智能体（通过 handoff 工具调用，名称必须完全一致）",
    ]
    if handoff_meta:
        for node, desc in handoff_meta:
            lines.append(f"- {node}：{desc}")
    else:
        for factory_name in registry.get_agent_names():
            node = registry_agent_to_node_name(factory_name)
            desc = registry.agents.get(factory_name, {}).get("description", factory_name)
            lines.append(f"- {node}：{desc}")
    lines.extend(
        [
            "",
            "## 规则",
            "1. 严格匹配用户请求范围，不要擅自扩展未询问的内容",
            "2. handoff 时给出完整、独立的子任务指令",
            "3. 子智能体返回后若已满足用户请求，直接输出最终答案",
            "4. 禁止输出调度过程话术（Transferring / handoff 等）",
            "5. 使用中文，语气专业友好",
        ]
    )
    return "\n".join(lines)


def build_handoff_tools(handoff_meta: Sequence[Tuple[str, str]]) -> List[Any]:
    _require_supervisor_deps()
    from langgraph_supervisor.handoff import create_handoff_tool

    return [
        create_handoff_tool(
            agent_name=node_name,
            description=f"交给 {desc} 处理",
        )
        for node_name, desc in handoff_meta
    ]


def build_supervisor_app(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    *,
    supervisor_prompt: str,
    transport: AgentTransport = TRANSPORT_LOCAL,
    a2a_endpoints: Sequence[A2AEndpoint] = (),
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Any:
    _require_supervisor_deps()
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph_supervisor.supervisor import create_supervisor

    sub_graphs, handoff_meta = resolve_supervisor_subgraphs(
        registry,
        transport=transport,
        a2a_endpoints=a2a_endpoints,
    )

    supervisor = create_supervisor(
        agents=sub_graphs,
        model=llm,
        tools=build_handoff_tools(handoff_meta),
        prompt=supervisor_prompt,
        supervisor_name="supervisor",
        output_mode="full_history",
    )
    if checkpointer is None:
        checkpointer = MemorySaver()
    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return supervisor.compile(**compile_kwargs)
