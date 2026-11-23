"""Rule-based scoring for travel task decomposition outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agent_framework.domain.parsing import parse_decomposition_response
from agent_framework.optimization.routing_assignment import routing_assignment_ratio

from .fixtures import DecompositionExpect, DependencyExpect, RoutingExpect


@dataclass
class DecompositionScore:
    total: float
    format_ok: bool
    subtask_count_ok: bool
    slot_ok: bool
    keyword_ok: bool
    forbidden_ok: bool
    dependency_ok: bool
    routing_assignment_ok: bool
    agent_coverage_ok: bool
    subtask_count: int
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "format_ok": self.format_ok,
            "subtask_count_ok": self.subtask_count_ok,
            "slot_ok": self.slot_ok,
            "keyword_ok": self.keyword_ok,
            "forbidden_ok": self.forbidden_ok,
            "dependency_ok": self.dependency_ok,
            "routing_assignment_ok": self.routing_assignment_ok,
            "agent_coverage_ok": self.agent_coverage_ok,
            "subtask_count": self.subtask_count,
            "details": list(self.details),
        }


def _normalize_parsed(
    result: Union[str, Dict[str, Any]],
    *,
    lang: str = "zh",
) -> Dict[str, Any]:
    if isinstance(result, dict):
        return {
            "totalGoal": str(result.get("totalGoal") or "").strip(),
            "subSteps": [str(item).strip() for item in result.get("subSteps") or [] if str(item).strip()],
        }
    return parse_decomposition_response(result, lang=lang)


def _active_subtasks(sub_steps: List[str]) -> List[str]:
    return [step for step in sub_steps if step and step.upper() != "NULL"]


def _score_slot_groups(combined_text: str, slot_groups: List[List[str]]) -> tuple[float, bool, List[str]]:
    if not slot_groups:
        return 1.0, True, []

    matched_groups = 0
    missing: List[str] = []
    for group in slot_groups:
        if any(token in combined_text for token in group):
            matched_groups += 1
        else:
            missing.append("/".join(group))

    ratio = matched_groups / len(slot_groups)
    return ratio, ratio >= 1.0, missing


def _normalize_depends_map(depends_map: Dict[str, List[str]]) -> Dict[str, tuple[str, ...]]:
    return {
        str(task_id): tuple(sorted(str(item) for item in deps if str(item).strip()))
        for task_id, deps in depends_map.items()
    }


def _score_dependency(
    execution_order: List[str],
    depends_map: Dict[str, List[str]],
    expect: Optional[DependencyExpect],
) -> tuple[float, bool, List[str]]:
    if expect is None:
        return 1.0, True, []

    details: List[str] = []
    checks = 0
    passed = 0

    actual_depends = _normalize_depends_map(depends_map)
    expected_depends = _normalize_depends_map(expect.depends_on)
    for task_id, expected_deps in expected_depends.items():
        checks += 1
        actual_deps = actual_depends.get(task_id, ())
        if actual_deps == expected_deps:
            passed += 1
        else:
            details.append(f"{task_id} 依赖期望 {list(expected_deps)}，实际 {list(actual_deps)}")

    if expect.execution_order:
        checks += 1
        if execution_order == expect.execution_order:
            passed += 1
        else:
            details.append(f"执行顺序期望 {expect.execution_order}，实际 {execution_order}")

    if checks == 0:
        return 1.0, True, []

    ratio = passed / checks
    return ratio, ratio >= 1.0, details


def score_decomposition(
    result: Union[str, Dict[str, Any]],
    expect: DecompositionExpect,
    *,
    routed_subtasks: Optional[List[Dict[str, Any]]] = None,
    routed_agents: Optional[List[str]] = None,
    execution_order: Optional[List[str]] = None,
    depends_map: Optional[Dict[str, List[str]]] = None,
    expect_routing: Optional[RoutingExpect] = None,
    expect_dependency: Optional[DependencyExpect] = None,
    lang: str = "zh",
) -> DecompositionScore:
    """Score a decomposition output against fixture expectations."""
    parsed = _normalize_parsed(result, lang=lang)
    sub_steps = _active_subtasks(parsed["subSteps"])
    combined_text = " ".join([parsed["totalGoal"], *sub_steps])
    details: List[str] = []
    total = 0.0

    format_ok = bool(parsed["totalGoal"]) and bool(sub_steps)
    if format_ok:
        total += 0.10
    else:
        details.append("缺少有效目标或非 NULL 子任务")

    subtask_count = len(sub_steps)
    subtask_count_ok = expect.min_subtasks <= subtask_count <= expect.max_subtasks
    if subtask_count_ok:
        total += 0.15
    else:
        details.append(
            f"子任务数 {subtask_count} 不在 [{expect.min_subtasks}, {expect.max_subtasks}]"
        )

    slot_ratio, slot_ok, missing_slots = _score_slot_groups(
        combined_text,
        expect.required_slot_groups,
    )
    total += 0.20 * slot_ratio
    if not slot_ok:
        details.append(f"缺少槽位组: {missing_slots}")
    keyword_ok = slot_ok

    forbidden_hits = [kw for kw in expect.forbidden_keywords if kw in combined_text]
    forbidden_ok = not forbidden_hits
    if forbidden_ok:
        total += 0.10
    else:
        details.append(f"出现禁止关键词: {forbidden_hits}")

    dep_ratio, dependency_ok, dep_details = _score_dependency(
        execution_order or [f"T{i + 1}" for i in range(subtask_count)],
        depends_map or {},
        expect_dependency,
    )
    total += 0.10 * dep_ratio
    if dep_details:
        details.extend(dep_details)

    routing_assignment_ok = True
    if expect_routing and routed_subtasks is not None:
        assign_ratio, routing_assignment_ok, routing_details = routing_assignment_ratio(
            routed_subtasks,
            expect_routing,
        )
        total += 0.25 * assign_ratio
        details.extend(routing_details)
    else:
        total += 0.25

    agent_coverage_ok = True
    if expect.mappable_agents:
        if routed_agents is None and routed_subtasks is not None:
            routed_agents = [
                str(item.get("agent") or "").strip()
                for item in routed_subtasks
                if item.get("agent")
            ]
        if routed_agents is None:
            details.append("未提供 LLM 路由结果，跳过 Agent 覆盖检查")
            total += 0.10
        else:
            covered = {agent for agent in routed_agents if agent}
            missing_agents = [name for name in expect.mappable_agents if name not in covered]
            agent_coverage_ok = not missing_agents
            if agent_coverage_ok:
                total += 0.10
            else:
                ratio = (len(expect.mappable_agents) - len(missing_agents)) / len(expect.mappable_agents)
                total += 0.10 * max(0.0, ratio)
                details.append(f"未覆盖 Agent（LLM 路由）: {missing_agents}")
    else:
        total += 0.10

    return DecompositionScore(
        total=min(total, 1.0),
        format_ok=format_ok,
        subtask_count_ok=subtask_count_ok,
        slot_ok=slot_ok,
        keyword_ok=keyword_ok,
        forbidden_ok=forbidden_ok,
        dependency_ok=dependency_ok,
        routing_assignment_ok=routing_assignment_ok,
        agent_coverage_ok=agent_coverage_ok,
        subtask_count=subtask_count,
        details=details,
    )
