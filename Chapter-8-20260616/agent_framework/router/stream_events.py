"""Router 流式事件 builder。"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_framework.router.plan import AgentCandidate, RoutingPlan, RoutingStep


def router_stage_event(stage: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": f"router.{stage}", "stage": stage, "data": data}


def router_plan_event(plan: RoutingPlan) -> Dict[str, Any]:
    return {
        "type": "router.plan",
        "stage": "plan",
        "data": plan.to_dict(),
        "_plan_obj": plan,
    }


def candidates_payload(candidates: List[AgentCandidate]) -> List[Dict[str, Any]]:
    return [{"name": c.name, "score": c.score} for c in candidates]


def steps_payload(steps: List[RoutingStep]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": s.step_id,
            "description": s.description,
            "agent": s.agent,
            "depends_on": list(s.depends_on),
        }
        for s in steps
    ]
