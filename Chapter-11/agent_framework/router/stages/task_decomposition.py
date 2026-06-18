"""Router 内嵌任务拆解（workflow profile 时产出 RoutingStep）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.router.config import RouterConfig
from agent_framework.router.plan import AgentCandidate, RoutingStep
from agent_framework.router.prompts.loader import get_task_decomposition_prompts
from agent_framework.router.stages.classification import build_agent_catalog
from agent_framework.tracing.trace_provider import span_name, trace_span


def parse_decomposition_response(raw: str, *, locale: str = "zh") -> Tuple[str, List[str]]:
    prompts = get_task_decomposition_prompts(locale)
    goal_key = prompts["keyword_goal"]
    subtask_key = prompts["keyword_subtasks"]
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    split_index = next(
        (idx for idx, line in enumerate(lines) if line.startswith(subtask_key)),
        len(lines),
    )
    goal_lines: List[str] = []
    for line in lines[:split_index]:
        if line.startswith(goal_key):
            goal_lines.append(line[len(goal_key) :].strip())
        elif line:
            goal_lines.append(line)
    total_goal = " ".join(goal_lines).strip()
    sub_steps: List[str] = []
    for line in lines[split_index + 1 :]:
        if line.startswith("-"):
            task = line.lstrip("-").strip()
            if task and task.upper() != "NULL":
                sub_steps.append(task)
    return total_goal, sub_steps


def build_routing_steps(
    sub_steps: List[str],
    candidates: List[AgentCandidate],
) -> List[RoutingStep]:
    strong = [c for c in candidates if c.name.lower() != "other" and c.score >= 0.5]
    steps: List[RoutingStep] = []
    for idx, desc in enumerate(sub_steps, start=1):
        agent = strong[idx - 1].name if idx - 1 < len(strong) else None
        steps.append(RoutingStep(step_id=f"T{idx}", description=desc, agent=agent))
    return steps


@trace_span(
    name=span_name("router.task_decomposition"),
    attrs_args=["query"],
    record_result=False,
)
async def run_task_decomposition(
    llm: ChatOpenAI,
    registry: SubAgentRegistry,
    query: str,
    candidates: List[AgentCandidate],
    *,
    locale: str = "zh",
    background_info: str = "",
    domain: str = "",
    config: RouterConfig | None = None,
    router_pre_survey: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[RoutingStep]]:
    cfg = config or RouterConfig()
    from agent_framework.router.stages.semantic_routing import (
        build_semantic_routing_steps,
        run_domain_decomposition,
        should_use_semantic_routing,
    )

    if should_use_semantic_routing(domain, cfg):
        total_goal, sub_steps = await run_domain_decomposition(
            llm,
            registry,
            domain,
            query,
            locale=locale,
            pre_survey=router_pre_survey,
        )
        steps = await build_semantic_routing_steps(
            llm,
            registry,
            domain,
            sub_steps,
            locale=locale,
        )
        return total_goal, steps

    prompts = get_task_decomposition_prompts(locale)
    agent_team = build_agent_catalog(registry, locale=locale)
    prompt = prompts["prompt"].format(
        background_info or "无",
        agent_team,
        query.strip(),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    total_goal, sub_steps = parse_decomposition_response(
        str(response.content or ""),
        locale=locale,
    )
    if not sub_steps:
        sub_steps = [query.strip()]
    steps = build_routing_steps(sub_steps, candidates)
    return total_goal, steps
