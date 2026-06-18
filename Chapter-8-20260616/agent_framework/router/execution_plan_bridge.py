"""RoutingPlan.steps → FixedGraph execution_plan 转换。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_framework.router.helpers import select_primary_candidate
from agent_framework.router.plan import RoutingPlan, RoutingStep
from agent_framework.router.pre_survey_bridge import pre_survey_from_routing_plan


def routing_steps_to_execution_plan(
    steps: List[RoutingStep],
    *,
    total_goal: str = "",
    user_query: str = "",
    pre_survey: Optional[Dict[str, Any]] = None,
    retrieved_memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """将 Router L1 拆解步骤转为 TaskPlanner / FixedGraph 可消费的 execution_plan。"""
    subtasks: List[Dict[str, Any]] = []
    execution_order: List[str] = []
    for idx, step in enumerate(steps, start=1):
        task_id = (step.step_id or f"T{idx}").strip() or f"T{idx}"
        execution_order.append(task_id)
        depends_on = (
            list(step.depends_on)
            if step.depends_on
            else ([execution_order[idx - 2]] if idx > 1 else [])
        )
        routing_status = "router_prefill" if step.agent else "pending"
        subtasks.append(
            {
                "task_id": task_id,
                "description": step.description,
                "agent": step.agent,
                "routing_status": routing_status,
                "params": {},
                "depends_on": depends_on,
            }
        )
    goal = (total_goal or user_query or "").strip()
    if not goal and subtasks:
        goal = subtasks[0]["description"]
    return {
        "pre_survey": dict(pre_survey or {}),
        "pre_survey_raw": "",
        "retrieved_memories": list(retrieved_memories or []),
        "total_goal": goal,
        "subtasks": subtasks,
        "execution_order": execution_order,
        "source": "router_engine",
        "pre_survey_source": (pre_survey or {}).get("source", "router_engine"),
    }


def enrich_execution_plan_pipeline_metadata(
    plan: Dict[str, Any],
    *,
    pre_survey_mode: str,
) -> Dict[str, Any]:
    """写入 pipeline 级 pre_survey 策略，便于 trace / API 观测。"""
    from agent_framework.domain.pipeline import normalize_pre_survey_mode

    enriched = dict(plan)
    enriched["pre_survey_mode"] = normalize_pre_survey_mode(pre_survey_mode)
    enriched.setdefault(
        "pre_survey_source",
        (enriched.get("pre_survey") or {}).get("source", "router_engine"),
    )
    return enriched


def execution_plan_from_routing_plan(
    plan: RoutingPlan,
    *,
    user_query: str = "",
    pre_survey: Optional[Dict[str, Any]] = None,
    retrieved_memories: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """当 RoutingPlan 含 steps 时生成预填 execution_plan，否则返回 None。"""
    if not plan.steps:
        return None
    router_pre_survey = pre_survey or pre_survey_from_routing_plan(plan)
    return routing_steps_to_execution_plan(
        plan.steps,
        total_goal=str(plan.metadata.get("decomposition_goal") or ""),
        user_query=user_query or plan.execution_query,
        pre_survey=router_pre_survey,
        retrieved_memories=retrieved_memories,
    )


def ensure_execution_plan_from_routing_plan(
    plan: RoutingPlan,
    *,
    user_query: str = "",
    pre_survey: Optional[Dict[str, Any]] = None,
    retrieved_memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """始终产出 execution_plan；无 steps 时合成单步计划。"""
    existing = execution_plan_from_routing_plan(
        plan,
        user_query=user_query,
        pre_survey=pre_survey,
        retrieved_memories=retrieved_memories,
    )
    if existing:
        return existing

    router_pre_survey = pre_survey or pre_survey_from_routing_plan(plan)
    query = (user_query or plan.execution_query or plan.rewritten_query).strip()
    primary = select_primary_candidate(plan.candidates)
    steps = [
        RoutingStep(
            "T1",
            query,
            primary.name if primary else None,
        )
    ]
    return routing_steps_to_execution_plan(
        steps,
        total_goal=str(plan.metadata.get("decomposition_goal") or query),
        user_query=query,
        pre_survey=router_pre_survey,
        retrieved_memories=retrieved_memories,
    )
