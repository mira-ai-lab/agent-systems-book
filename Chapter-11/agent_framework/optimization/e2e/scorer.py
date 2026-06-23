"""Rule-based scoring for travel end-to-end orchestration runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .expectations import E2eExpect

_COMPLETED_STATUSES = frozenset({"completed", "ok"})


@dataclass
class E2eScore:
    total: float
    response_ok: bool
    keyword_ok: bool
    forbidden_ok: bool
    agents_ok: bool
    completion_ok: bool
    completed_subtasks: int
    invoked_agents: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "response_ok": self.response_ok,
            "keyword_ok": self.keyword_ok,
            "forbidden_ok": self.forbidden_ok,
            "agents_ok": self.agents_ok,
            "completion_ok": self.completion_ok,
            "completed_subtasks": self.completed_subtasks,
            "invoked_agents": list(self.invoked_agents),
            "details": list(self.details),
        }


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _invoked_agents(subtask_results: Dict[str, Any]) -> Set[str]:
    agents: Set[str] = set()
    for item in (subtask_results or {}).values():
        if not isinstance(item, dict):
            continue
        agent = _normalize_text(item.get("agent"))
        if agent:
            agents.add(agent)
    return agents


def _completed_subtask_count(subtask_results: Dict[str, Any]) -> int:
    count = 0
    for item in (subtask_results or {}).values():
        if not isinstance(item, dict):
            continue
        status = _normalize_text(item.get("status")).lower()
        if status in _COMPLETED_STATUSES:
            count += 1
    return count


def _slot_groups_satisfied(text: str, slot_groups: List[List[str]]) -> tuple[bool, List[str]]:
    if not slot_groups:
        return True, []
    missing: List[str] = []
    for group in slot_groups:
        if not any(token in text for token in group):
            missing.append(" / ".join(group))
    return not missing, missing


def score_e2e_run(result: Dict[str, Any], expect: E2eExpect) -> E2eScore:
    """Score a full orchestration result against E2E expectations."""
    details: List[str] = []
    total = 0.0

    final_response = _normalize_text(result.get("final_response"))
    subtask_results = result.get("subtask_results") or {}
    invoked = sorted(_invoked_agents(subtask_results))
    completed = _completed_subtask_count(subtask_results)

    response_ok = bool(final_response) if expect.require_final_response else True
    if response_ok:
        total += 0.10
    else:
        details.append("缺少 final_response")

    keyword_ok = True
    if expect.required_response_slot_groups:
        keyword_ok, missing_groups = _slot_groups_satisfied(
            final_response,
            expect.required_response_slot_groups,
        )
        if keyword_ok:
            total += 0.25
        else:
            details.append(f"回复缺少关键词组: {missing_groups}")
    elif expect.required_response_keywords:
        missing_keywords = [kw for kw in expect.required_response_keywords if kw not in final_response]
        keyword_ok = not missing_keywords
        if keyword_ok:
            total += 0.25
        else:
            details.append(f"回复缺少关键词: {missing_keywords}")
    else:
        total += 0.25

    forbidden_hits = [kw for kw in expect.forbidden_response_keywords if kw in final_response]
    forbidden_ok = not forbidden_hits
    if forbidden_ok:
        total += 0.15
    else:
        details.append(f"回复出现禁止关键词: {forbidden_hits}")

    agents_ok = True
    if expect.required_agents:
        missing_agents = [name for name in expect.required_agents if name not in invoked]
        agents_ok = not missing_agents
        if agents_ok:
            total += 0.35
        else:
            matched = len(expect.required_agents) - len(missing_agents)
            ratio = matched / len(expect.required_agents)
            total += 0.35 * max(0.0, ratio)
            details.append(f"未调用期望 Agent: {missing_agents}")
    else:
        total += 0.35

    completion_ok = completed >= expect.min_completed_subtasks
    if completion_ok:
        total += 0.15
    else:
        details.append(
            f"完成子任务 {completed} < 期望最少 {expect.min_completed_subtasks}"
        )

    return E2eScore(
        total=min(total, 1.0),
        response_ok=response_ok,
        keyword_ok=keyword_ok,
        forbidden_ok=forbidden_ok,
        agents_ok=agents_ok,
        completion_ok=completion_ok,
        completed_subtasks=completed,
        invoked_agents=invoked,
        details=details,
    )
