"""Batch evaluator for travel agent routing benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.decomposition.fixtures import (
    DecompositionBenchmarkCase,
    DecompositionFixtures,
    load_decomposition_fixtures,
)

from .scorer import RoutingScore, score_routing


@dataclass
class RoutingCaseResult:
    case_id: str
    query: str
    score: RoutingScore
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "score": self.score.to_dict(),
            "subtasks": list(self.subtasks),
            "raw_output": self.raw_output,
        }


@dataclass
class RoutingBenchmarkReport:
    domain: str
    locale: str
    split: str
    case_count: int
    average_score: float
    cases: List[RoutingCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "locale": self.locale,
            "split": self.split,
            "case_count": self.case_count,
            "average_score": round(self.average_score, 4),
            "cases": [item.to_dict() for item in self.cases],
        }


async def evaluate_routing_case(
    planner: TaskPlanner,
    case: DecompositionBenchmarkCase,
) -> RoutingCaseResult:
    if not case.routing_input or not case.expect_routing:
        raise ValueError(f"case {case.case_id} 缺少 routing benchmark 字段")

    routing_input = case.routing_input
    subtasks = await planner.route_to_agents(
        routing_input.sub_steps,
        routing_input.execution_order,
        routing_input.depends_map,
    )
    score = score_routing(subtasks, case.expect_routing)
    return RoutingCaseResult(
        case_id=case.case_id,
        query=case.query,
        score=score,
        subtasks=subtasks,
        raw_output=json.dumps(subtasks, ensure_ascii=False, indent=2),
    )


async def evaluate_routing_benchmark(
    planner: TaskPlanner,
    *,
    fixtures: Optional[DecompositionFixtures] = None,
    split: str = "dev",
) -> RoutingBenchmarkReport:
    loaded = fixtures or load_decomposition_fixtures()
    selected = loaded.routing_cases_for_split(split)

    case_results: List[RoutingCaseResult] = []
    for case in selected:
        case_results.append(await evaluate_routing_case(planner, case))

    average_score = (
        sum(item.score.total for item in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    return RoutingBenchmarkReport(
        domain=loaded.domain,
        locale=loaded.locale,
        split=split,
        case_count=len(case_results),
        average_score=average_score,
        cases=case_results,
    )
