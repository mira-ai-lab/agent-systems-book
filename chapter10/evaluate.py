"""Rule-based benchmark scoring and evaluation.

Scores decompose the Chapter-10 self-evolution loop into two independent
0~1 metrics:

* L1 (decomposition): subtask structure, slot coverage, and constraint
  retention.
* L2 (routing): subtask-to-agent matching plus tool-parameter completeness.

Both metrics are normalised independently, so ``total = (L1 + L2) / 2``.
The ``metric`` argument selects which evaluation to run, which lets the
optimizer tune the decomposition slot against L1 and the routing slot
against L2.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from runtime import Planner


@dataclass
class CaseScore:
    case_id: str
    total: float
    l1: Optional[float] = None
    l2: Optional[float] = None
    format_ok: bool = True
    subtask_count_ok: bool = True
    slots_ok: bool = True
    forbidden_ok: bool = True
    routing_ok: bool = True
    agent_coverage_ok: bool = True
    params_ok: bool = True
    subtask_count: int = 0
    details: List[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    split: str
    metric: str
    case_count: int
    average_score: float
    cases: List[CaseScore] = field(default_factory=list)

    @property
    def average_l1(self) -> Optional[float]:
        values = [c.l1 for c in self.cases if c.l1 is not None]
        return sum(values) / len(values) if values else None

    @property
    def average_l2(self) -> Optional[float]:
        values = [c.l2 for c in self.cases if c.l2 is not None]
        return sum(values) / len(values) if values else None


def load_fixtures(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_for_split(fixtures: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    ids = fixtures.get("splits", {}).get(split, [])
    by_id = {str(case["id"]): case for case in fixtures.get("cases", [])}
    return [by_id[i] for i in ids if i in by_id]


def _slots_ok(combined_text: str, slot_groups: List[List[str]]) -> tuple[bool, List[str]]:
    if not slot_groups:
        return True, []
    missing = [
        "/".join(group)
        for group in slot_groups
        if not any(token in combined_text for token in group)
    ]
    return not missing, missing


def _background_from_case(case: Dict[str, Any]) -> str:
    pre_survey = case.get("pre_survey") or {}
    return "【思维链预调查】\n" + json.dumps(
        {k: v for k, v in pre_survey.items() if k != "raw_text"},
        ensure_ascii=False,
    )


def _valid_sub_steps(parsed: Dict[str, Any]) -> List[str]:
    return [
        s for s in (parsed.get("sub_steps") or [])
        if s and s.strip().upper() != "NULL"
    ]


def _score_l1(
    parsed: Dict[str, Any],
    sub_steps: List[str],
    case: Dict[str, Any],
) -> tuple[float, Dict[str, bool], List[str]]:
    """Score decomposition quality (L1) into a 0~1 value.

    四个子项：格式 0.10、子任务数 0.15、槽位 0.25、禁止词 0.10，合计 0.60，
    再除以 0.60 归一化。槽位用 ``required_slot_groups``，缺省回退到
    ``required_keywords``。
    """
    expect = case.get("expect") or {}
    total_goal = str(parsed.get("total_goal") or "").strip()
    combined = " ".join([total_goal, *sub_steps])
    details: List[str] = []

    format_ok = bool(total_goal) and bool(sub_steps)
    format_raw = 0.10 if format_ok else 0.0
    if not format_ok:
        details.append("缺少目标或有效子任务")

    n = len(sub_steps)
    lo = int(expect.get("min_subtasks", 1))
    hi = int(expect.get("max_subtasks", 99))
    subtask_count_ok = lo <= n <= hi
    subtask_raw = 0.15 if subtask_count_ok else 0.0
    if not subtask_count_ok:
        details.append(f"子任务数 {n} 不在 [{lo}, {hi}]")

    groups = [[str(t) for t in g] for g in (expect.get("required_slot_groups") or [])]
    if not groups:
        groups = [[str(k)] for k in (expect.get("required_keywords") or [])]
    slots_ok, missing = _slots_ok(combined, groups)
    slots_raw = 0.25 if (not groups or slots_ok) else 0.0
    if groups and not slots_ok:
        details.append(f"缺少槽位: {missing}")

    forbidden = [k for k in (expect.get("forbidden_keywords") or []) if k in combined]
    forbidden_ok = not forbidden
    forbidden_raw = 0.10 if forbidden_ok else 0.0
    if not forbidden_ok:
        details.append(f"出现禁止词: {forbidden}")

    l1 = (format_raw + subtask_raw + slots_raw + forbidden_raw) / 0.60
    flags = {
        "format_ok": format_ok,
        "subtask_count_ok": subtask_count_ok,
        "slots_ok": slots_ok,
        "forbidden_ok": forbidden_ok,
    }
    return l1, flags, details


def _score_l2(
    records: List[Dict[str, Any]],
    case: Dict[str, Any],
) -> tuple[float, Dict[str, bool], List[str]]:
    """Score routing quality (L2) into a 0~1 value.

    三个子项：Agent 匹配 0.25、Agent 覆盖 0.15、参数完整性 0.20，合计 0.60，
    再除以 0.60 归一化。用例未声明某项期望时，该项给满分。
    """
    expect = case.get("expect") or {}
    expect_routing = case.get("expect_routing") or {}
    assignments = expect_routing.get("assignments") or []
    agent_by_id = {r.get("task_id"): r.get("agent") or "" for r in records}
    params_by_id = {r.get("task_id"): r.get("params") or {} for r in records}
    details: List[str] = []

    routing_ok = True
    routing_raw = 0.0
    if assignments:
        correct = 0
        for a in assignments:
            tid = str(a.get("task_id"))
            expected = str(a.get("expected_agent"))
            if agent_by_id.get(tid) == expected:
                correct += 1
            else:
                routing_ok = False
                details.append(
                    f"{tid} 期望 {expected}，实际 {agent_by_id.get(tid) or '未路由'}"
                )
        routing_raw = 0.25 * (correct / len(assignments))
    else:
        routing_raw = 0.25

    mappable = [str(m) for m in (expect.get("mappable_agents") or [])]
    agent_coverage_ok = True
    coverage_raw = 0.0
    if mappable:
        covered = {a for a in agent_by_id.values() if a}
        missing = [m for m in mappable if m not in covered]
        agent_coverage_ok = not missing
        coverage_raw = 0.15 * max(0.0, (len(mappable) - len(missing)) / len(mappable))
        if missing:
            details.append(f"未覆盖 Agent: {missing}")
    else:
        coverage_raw = 0.15

    tool_checks = expect.get("tool_checks") or []
    params_ok = True
    params_raw = 0.0
    if tool_checks:
        total_checks = 0
        passed = 0
        for check in tool_checks:
            tid = str(check.get("task_id"))
            field_contains = check.get("field_contains") or {}
            params = params_by_id.get(tid) or {}
            for field, expected_values in field_contains.items():
                total_checks += 1
                actual = params.get(field)
                if actual is not None and any(
                    str(ev) in str(actual) for ev in expected_values
                ):
                    passed += 1
                else:
                    params_ok = False
                    details.append(f"{tid}.params.{field} 缺少 {expected_values}")
        params_raw = 0.20 * (passed / total_checks) if total_checks else 0.20
    else:
        params_raw = 0.20

    l2 = (routing_raw + coverage_raw + params_raw) / 0.60
    flags = {
        "routing_ok": routing_ok,
        "agent_coverage_ok": agent_coverage_ok,
        "params_ok": params_ok,
    }
    return l2, flags, details


def score_case(
    parsed: Dict[str, Any],
    sub_steps: List[str],
    records: List[Dict[str, Any]],
    case: Dict[str, Any],
) -> CaseScore:
    """同时计算 L1 与 L2，总分取二者平均（objective=l1_l2）。"""
    l1, flags1, details1 = _score_l1(parsed, sub_steps, case)
    l2, flags2, details2 = _score_l2(records, case)
    return CaseScore(
        case_id=str(case.get("id")),
        total=(l1 + l2) / 2,
        l1=l1,
        l2=l2,
        subtask_count=len(sub_steps),
        details=[*details1, *details2],
        **flags1,
        **flags2,
    )


def score_decomposition(
    parsed: Dict[str, Any],
    sub_steps: List[str],
    case: Dict[str, Any],
) -> CaseScore:
    l1, flags, details = _score_l1(parsed, sub_steps, case)
    return CaseScore(
        case_id=str(case.get("id")),
        total=l1,
        l1=l1,
        subtask_count=len(sub_steps),
        details=details,
        **flags,
    )


def score_routing(records: List[Dict[str, Any]], case: Dict[str, Any]) -> CaseScore:
    l2, flags, details = _score_l2(records, case)
    return CaseScore(
        case_id=str(case.get("id")),
        total=l2,
        l2=l2,
        details=details,
        **flags,
    )


async def evaluate_case(
    planner: Planner,
    case: Dict[str, Any],
    *,
    metric: str = "l1_l2",
) -> CaseScore:
    """Evaluate one case under ``metric`` (``l1`` / ``l2`` / ``l1_l2``)."""
    pre_survey = case.get("pre_survey") or {}

    if metric == "l1":
        parsed = await planner.decompose(
            case["query"],
            background_info=_background_from_case(case),
        )
        return score_decomposition(parsed, _valid_sub_steps(parsed), case)

    if metric == "l2":
        routing_input = case.get("routing_input") or {}
        sub_steps = [s for s in (routing_input.get("sub_steps") or []) if s]
        if sub_steps:
            execution_order = routing_input.get("execution_order") or [
                f"T{i + 1}" for i in range(len(sub_steps))
            ]
            depends_map = routing_input.get("depends_map") or {
                tid: [] for tid in execution_order
            }
        else:
            parsed = await planner.decompose(
                case["query"],
                background_info=_background_from_case(case),
            )
            sub_steps = _valid_sub_steps(parsed)
            execution_order, depends_map = await planner.dependency_analysis(sub_steps)

        records = await planner.route_detailed(
            sub_steps,
            execution_order,
            depends_map,
            pre_survey,
        )
        return score_routing(records, case)

    parsed = await planner.decompose(
        case["query"],
        background_info=_background_from_case(case),
    )
    sub_steps = _valid_sub_steps(parsed)
    execution_order, depends_map = await planner.dependency_analysis(sub_steps)
    records = await planner.route_detailed(
        sub_steps,
        execution_order,
        depends_map,
        pre_survey,
    )
    return score_case(parsed, sub_steps, records, case)


async def evaluate_benchmark(
    planner: Planner,
    fixtures: Dict[str, Any],
    split: str,
    *,
    metric: str = "l1_l2",
) -> BenchmarkReport:
    cases = cases_for_split(fixtures, split)
    results = [await evaluate_case(planner, case, metric=metric) for case in cases]
    average = sum(r.total for r in results) / len(results) if results else 0.0
    return BenchmarkReport(
        split=split,
        metric=metric,
        case_count=len(results),
        average_score=average,
        cases=results,
    )


async def collect_failures(
    planner: Planner,
    cases: List[Dict[str, Any]],
    failure_threshold: float,
    metric: str = "l1_l2",
) -> List[Tuple[Any, Dict[str, Any]]]:
    """Return ``(score, case)`` pairs whose score is below the threshold."""
    failures: List[Tuple[Any, Dict[str, Any]]] = []
    for case in cases:
        result = await evaluate_case(planner, case, metric=metric)
        if result.total < failure_threshold:
            failures.append((result, case))
    return failures


async def average_case_score(
    planner: Planner,
    cases: List[Dict[str, Any]],
    metric: str = "l1_l2",
) -> float:
    """Return the mean ``metric`` score across ``cases``."""
    results = [await evaluate_case(planner, case, metric=metric) for case in cases]
    return sum(result.total for result in results) / len(results) if results else 0.0
