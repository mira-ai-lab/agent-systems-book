"""Shared routing assignment scoring helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from agent_framework.optimization.decomposition.fixtures import RoutingExpect


def routing_assignment_ratio(
    subtasks: List[Dict[str, Any]],
    expect: RoutingExpect,
) -> tuple[float, bool, List[str]]:
    """Return (ratio, all_match, details) for agent assignment checks."""
    details: List[str] = []
    if not expect.assignments:
        return 1.0, True, details

    by_id = {str(item.get("task_id")): item for item in subtasks if item.get("task_id")}
    expected = expect.assignments

    if len(subtasks) == len(expected):
        matched = 0
        for assignment in expected:
            item = by_id.get(assignment.task_id)
            if item and item.get("agent") == assignment.expected_agent:
                matched += 1
            else:
                actual = item.get("agent") if item else None
                details.append(
                    f"{assignment.task_id} 期望 {assignment.expected_agent}，实际 {actual}"
                )
        ratio = matched / len(expected)
        return ratio, ratio >= 1.0, details

    expected_agents = [assignment.expected_agent for assignment in expected]
    actual_agents = [str(item.get("agent") or "") for item in subtasks if item.get("agent")]
    if not actual_agents:
        details.append("路由结果为空")
        return 0.0, False, details

    remaining = list(actual_agents)
    matched = 0
    missing: List[str] = []
    for agent in expected_agents:
        if agent in remaining:
            remaining.remove(agent)
            matched += 1
        else:
            missing.append(agent)
    ratio = matched / len(expected_agents)
    if missing:
        details.append(f"子任务数不一致({len(subtasks)} vs {len(expected)})，未匹配 Agent: {missing}")
    return ratio, ratio >= 1.0 and not missing, details
