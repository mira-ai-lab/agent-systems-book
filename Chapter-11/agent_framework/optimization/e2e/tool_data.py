"""Structured ``tool_data`` checks for E2E benchmark scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent_framework.optimization.decomposition.fixtures import ToolDataCheck

_COMPLETED_STATUSES = frozenset({"completed", "ok"})


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def iter_tool_payloads(tool_data: Any) -> List[Dict[str, Any]]:
    """Expand single tool output or ``{calls: [...]}`` into inspectable dicts."""
    if not isinstance(tool_data, dict):
        return []
    calls = tool_data.get("calls")
    if isinstance(calls, list):
        payloads = [item for item in calls if isinstance(item, dict)]
        return payloads or [tool_data]
    return [tool_data]


def score_tool_data_checks(
    subtask_results: Dict[str, Any],
    checks: List[ToolDataCheck],
) -> Tuple[bool, float, List[str]]:
    """Score configured tool_data field checks. Returns (ok, partial_credit, details)."""
    if not checks:
        return True, 1.0, []

    details: List[str] = []
    passed = 0
    for check in checks:
        item = subtask_results.get(check.task_id)
        if not isinstance(item, dict):
            details.append(f"tool_check {check.task_id}: 无 subtask 结果")
            continue

        status = _normalize_text(item.get("status")).lower()
        if status not in _COMPLETED_STATUSES:
            details.append(f"tool_check {check.task_id}: 子任务未完成 (status={status or 'missing'})")
            continue

        payloads = iter_tool_payloads(item.get("tool_data"))
        if not payloads:
            details.append(f"tool_check {check.task_id}: 缺少 tool_data")
            continue

        if check.forbid_error and any(
            isinstance(payload, dict) and payload.get("error") for payload in payloads
        ):
            details.append(f"tool_check {check.task_id}: tool_data 含 error")
            continue

        check_ok = True
        for field, tokens in check.field_contains.items():
            field_ok = any(
                any(token in _normalize_text(payload.get(field)) for token in tokens)
                for payload in payloads
            )
            if not field_ok:
                check_ok = False
                details.append(
                    f"tool_check {check.task_id}: 字段 {field} 未匹配 {tokens}"
                )
        if check_ok:
            passed += 1

    ratio = passed / len(checks)
    return ratio >= 1.0, ratio, details
