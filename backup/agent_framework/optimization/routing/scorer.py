"""Rule-based scoring for agent routing outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent_framework.optimization.decomposition.fixtures import RoutingExpect

from agent_framework.optimization.routing_assignment import routing_assignment_ratio


@dataclass
class RoutingScore:
    total: float
    coverage_ok: bool
    agent_match_ok: bool
    routing_status_ok: bool
    params_ok: bool
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "coverage_ok": self.coverage_ok,
            "agent_match_ok": self.agent_match_ok,
            "routing_status_ok": self.routing_status_ok,
            "params_ok": self.params_ok,
            "details": list(self.details),
        }


def score_routing(subtasks: List[Dict[str, Any]], expect: RoutingExpect) -> RoutingScore:
    details: List[str] = []
    total = 0.0
    by_id = {str(item.get("task_id")): item for item in subtasks if item.get("task_id")}

    expected_ids = [item.task_id for item in expect.assignments]
    coverage_ok = all(task_id in by_id for task_id in expected_ids)
    if coverage_ok:
        total += 0.2
    else:
        missing = [task_id for task_id in expected_ids if task_id not in by_id]
        details.append(f"缺少子任务路由: {missing}")

    assign_ratio, agent_match_ok, assign_details = routing_assignment_ratio(subtasks, expect)
    details.extend(assign_details)
    if agent_match_ok:
        total += 0.5
    else:
        total += 0.5 * assign_ratio

    bad_status = [
        str(item.get("task_id"))
        for item in by_id.values()
        if str(item.get("routing_status") or "") == "routing_failed"
    ]
    routing_status_ok = not bad_status
    if routing_status_ok:
        total += 0.15
    else:
        details.append(f"路由失败: {bad_status}")

    params_missing = [
        str(item.get("task_id"))
        for item in by_id.values()
        if not isinstance(item.get("params"), dict) or not item.get("params")
    ]
    params_ok = not params_missing
    if params_ok:
        total += 0.15
    else:
        details.append(f"params 为空: {params_missing}")

    return RoutingScore(
        total=min(total, 1.0),
        coverage_ok=coverage_ok,
        agent_match_ok=agent_match_ok,
        routing_status_ok=routing_status_ok,
        params_ok=params_ok,
        details=details,
    )
