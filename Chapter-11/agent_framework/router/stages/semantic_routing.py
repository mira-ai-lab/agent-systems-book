"""领域语义拆解与 Agent 路由（Router workflow profile 增强）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.router.config import RouterConfig
from agent_framework.router.plan import RoutingStep
from agent_framework.tracing.trace_provider import span_name, trace_span

SEMANTIC_ROUTING_DOMAINS = frozenset({"travel"})


def should_use_semantic_routing(domain: str, config: RouterConfig) -> bool:
    """当领域插件提供 TaskPlanner 话术时，用 agent_routing 替代 classification 顺位绑 Agent。"""
    return bool(
        config.semantic_task_routing
        and (domain or "").strip() in SEMANTIC_ROUTING_DOMAINS
    )


def create_domain_task_planner(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    domain: str,
    *,
    locale: str = "zh",
    enable_guess_agent: bool = True,
) -> TaskPlanner:
    from agent_framework.domain.plugin_registry import get_domain_plugin

    plugin = get_domain_plugin(domain)
    prompts = plugin.create_prompts(locale=locale)
    domain_config = plugin.create_domain_config(enable_guess_agent=enable_guess_agent)
    return TaskPlanner(llm, registry, prompts, domain_config)


@trace_span(
    name=span_name("router.domain_decomposition"),
    attrs_args=["domain", "query"],
    record_result=False,
)
async def run_domain_decomposition(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    domain: str,
    query: str,
    *,
    locale: str = "zh",
    pre_survey: Optional[Dict[str, Any]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[str]]:
    """使用领域 decomposition_prompt 拆解子任务（可注入 Router bridge pre_survey）。"""
    planner = create_domain_task_planner(
        llm,
        registry,
        domain,
        locale=locale,
        enable_guess_agent=True,
    )
    decomp = await planner.run_decomposition(
        query,
        dict(pre_survey or {}),
        list(memories or []),
    )
    sub_steps = [
        step.strip()
        for step in decomp.get("subSteps", [])
        if step and str(step).strip().upper() != "NULL"
    ]
    if not sub_steps:
        sub_steps = [query.strip()]
    return str(decomp.get("totalGoal") or "").strip(), sub_steps


@trace_span(
    name=span_name("router.semantic_routing"),
    attrs_args=["domain"],
    record_result=False,
)
async def build_semantic_routing_steps(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    domain: str,
    sub_steps: List[str],
    *,
    locale: str = "zh",
) -> List[RoutingStep]:
    """依赖分析 + agent_routing，产出带 depends_on 的 RoutingStep。"""
    planner = create_domain_task_planner(
        llm,
        registry,
        domain,
        locale=locale,
        enable_guess_agent=True,
    )
    execution_order, depends_map = await planner.run_dependency_analysis(sub_steps)
    subtasks = await planner.route_to_agents(sub_steps, execution_order, depends_map)
    return [
        RoutingStep(
            step_id=item["task_id"],
            description=item["description"],
            agent=item.get("agent"),
            depends_on=tuple(item.get("depends_on") or []),
        )
        for item in subtasks
    ]
