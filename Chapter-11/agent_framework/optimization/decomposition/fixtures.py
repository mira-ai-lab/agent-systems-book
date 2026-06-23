"""Travel decomposition + routing benchmark fixture loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_framework.config import PROJECT_ROOT

DEFAULT_FIXTURES_PATH = (
    PROJECT_ROOT / "data" / "benchmark" / "travel_decomposition" / "fixtures.json"
)
VALID_SPLITS = ("train", "dev", "test", "all")


@dataclass(frozen=True)
class DecompositionExpect:
    min_subtasks: int = 1
    max_subtasks: int = 99
    required_keywords: List[str] = field(default_factory=list)
    required_slot_groups: List[List[str]] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    mappable_agents: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DependencyExpect:
    """Expected dependency graph for live dependency analysis."""

    depends_on: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: Optional[List[str]] = None


@dataclass(frozen=True)
class RoutingInput:
    sub_steps: List[str]
    execution_order: List[str]
    depends_map: Dict[str, List[str]]


@dataclass(frozen=True)
class RoutingAssignmentExpect:
    task_id: str
    expected_agent: str


@dataclass(frozen=True)
class RoutingExpect:
    assignments: List[RoutingAssignmentExpect]


@dataclass(frozen=True)
class DecompositionBenchmarkCase:
    case_id: str
    query: str
    pre_survey: Dict[str, Any]
    expect: DecompositionExpect
    routing_input: Optional[RoutingInput] = None
    expect_routing: Optional[RoutingExpect] = None
    expect_dependency: Optional[DependencyExpect] = None


@dataclass(frozen=True)
class DecompositionFixtures:
    version: str
    domain: str
    locale: str
    cases: List[DecompositionBenchmarkCase]
    splits: Dict[str, List[str]]

    def cases_for_split(self, split: str) -> List[DecompositionBenchmarkCase]:
        normalized = (split or "all").strip().lower()
        if normalized not in VALID_SPLITS:
            raise ValueError(f"不支持的 split='{split}'，可选: {', '.join(VALID_SPLITS)}")
        if normalized == "all":
            return list(self.cases)
        ids = set(self.splits.get(normalized, []))
        if not ids:
            raise ValueError(f"fixtures 中未定义 split='{normalized}'")
        selected = [case for case in self.cases if case.case_id in ids]
        missing = ids - {case.case_id for case in selected}
        if missing:
            raise ValueError(f"split '{normalized}' 引用了未知 case: {sorted(missing)}")
        return selected

    def routing_cases_for_split(self, split: str) -> List[DecompositionBenchmarkCase]:
        selected = self.cases_for_split(split)
        missing = [case.case_id for case in selected if not case.routing_input or not case.expect_routing]
        if missing:
            raise ValueError(f"以下 case 缺少 routing_input/expect_routing: {missing}")
        return selected


def default_fixtures_path() -> Path:
    return DEFAULT_FIXTURES_PATH


def _parse_slot_groups(raw: Any) -> List[List[str]]:
    groups: List[List[str]] = []
    if not isinstance(raw, list):
        return groups
    for item in raw:
        if isinstance(item, list):
            tokens = [str(token).strip() for token in item if str(token).strip()]
            if tokens:
                groups.append(tokens)
        elif str(item).strip():
            groups.append([str(item).strip()])
    return groups


def _parse_expect(raw: Dict[str, Any]) -> DecompositionExpect:
    slot_groups = _parse_slot_groups(raw.get("required_slot_groups"))
    keywords = [str(item).strip() for item in raw.get("required_keywords", []) if str(item).strip()]
    if not slot_groups and keywords:
        slot_groups = [[keyword] for keyword in keywords]
    return DecompositionExpect(
        min_subtasks=int(raw.get("min_subtasks", 1)),
        max_subtasks=int(raw.get("max_subtasks", 99)),
        required_keywords=keywords,
        required_slot_groups=slot_groups,
        forbidden_keywords=[str(item).strip() for item in raw.get("forbidden_keywords", []) if str(item).strip()],
        mappable_agents=[str(item).strip() for item in raw.get("mappable_agents", []) if str(item).strip()],
    )


def _parse_dependency_expect(raw: Optional[Dict[str, Any]]) -> Optional[DependencyExpect]:
    if not raw:
        return None
    depends_raw = raw.get("depends_on") or {}
    depends_on = {
        str(key): [str(item) for item in (value or [])]
        for key, value in depends_raw.items()
    }
    order_raw = raw.get("execution_order")
    execution_order = (
        [str(item).strip() for item in order_raw if str(item).strip()]
        if isinstance(order_raw, list)
        else None
    )
    if not depends_on and not execution_order:
        return None
    return DependencyExpect(depends_on=depends_on, execution_order=execution_order)


def _parse_routing_input(raw: Optional[Dict[str, Any]]) -> Optional[RoutingInput]:
    if not raw:
        return None
    sub_steps = [str(item).strip() for item in raw.get("sub_steps") or [] if str(item).strip()]
    if not sub_steps:
        raise ValueError("routing_input.sub_steps 不能为空")
    execution_order = [str(item).strip() for item in raw.get("execution_order") or [] if str(item).strip()]
    if not execution_order:
        execution_order = [f"T{i + 1}" for i in range(len(sub_steps))]
    depends_map_raw = raw.get("depends_map") or {}
    depends_map = {
        str(key): [str(item) for item in (value or [])]
        for key, value in depends_map_raw.items()
    }
    return RoutingInput(sub_steps=sub_steps, execution_order=execution_order, depends_map=depends_map)


def _parse_routing_expect(raw: Optional[Dict[str, Any]]) -> Optional[RoutingExpect]:
    if not raw:
        return None
    assignments: List[RoutingAssignmentExpect] = []
    for idx, item in enumerate(raw.get("assignments") or []):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        expected_agent = str(item.get("expected_agent") or "").strip()
        if not task_id or not expected_agent:
            raise ValueError(f"expect_routing.assignments[{idx}] 缺少 task_id/expected_agent")
        assignments.append(RoutingAssignmentExpect(task_id=task_id, expected_agent=expected_agent))
    if not assignments:
        raise ValueError("expect_routing.assignments 不能为空")
    return RoutingExpect(assignments=assignments)


def load_decomposition_fixtures(path: Optional[Path] = None) -> DecompositionFixtures:
    fixture_path = path or default_fixtures_path()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    version = str(payload.get("version") or "1.0.0").strip() or "1.0.0"
    domain = str(payload.get("domain") or "").strip()
    if not domain:
        raise ValueError("fixtures 缺少 domain")

    locale = str(payload.get("locale") or "zh").strip() or "zh"
    raw_cases = payload.get("cases") or []
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("fixtures 缺少 cases")

    cases: List[DecompositionBenchmarkCase] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"benchmark case[{idx}] 缺少 query")
        case_id = str(item.get("id") or f"case-{idx + 1}").strip()
        pre_survey = item.get("pre_survey") or {}
        if not isinstance(pre_survey, dict):
            raise ValueError(f"benchmark case[{idx}] pre_survey 必须是对象")
        expect = _parse_expect(item.get("expect") or {})
        routing_input = _parse_routing_input(item.get("routing_input"))
        expect_routing = _parse_routing_expect(item.get("expect_routing"))
        expect_dependency = _parse_dependency_expect(item.get("expect_dependency"))
        cases.append(
            DecompositionBenchmarkCase(
                case_id=case_id,
                query=query,
                pre_survey=pre_survey,
                expect=expect,
                routing_input=routing_input,
                expect_routing=expect_routing,
                expect_dependency=expect_dependency,
            )
        )

    splits = payload.get("splits") or {}
    if not isinstance(splits, dict):
        raise ValueError("fixtures splits 必须是对象")

    return DecompositionFixtures(
        version=version,
        domain=domain,
        locale=locale,
        cases=cases,
        splits={str(key): list(value or []) for key, value in splits.items()},
    )
