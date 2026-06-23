"""Batch evaluator for travel task decomposition benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_framework.domain.task_planner import TaskPlanner

from .fixtures import DecompositionBenchmarkCase, DecompositionFixtures, load_decomposition_fixtures
from .scorer import DecompositionScore, score_decomposition


@dataclass
class DecompositionCaseResult:
    case_id: str
    query: str
    score: DecompositionScore
    total_goal: str = ""
    sub_steps: List[str] = field(default_factory=list)
    routed_agents: List[str] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    depends_map: Dict[str, List[str]] = field(default_factory=dict)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "score": self.score.to_dict(),
            "total_goal": self.total_goal,
            "sub_steps": list(self.sub_steps),
            "routed_agents": list(self.routed_agents),
            "execution_order": list(self.execution_order),
            "depends_map": {key: list(value) for key, value in self.depends_map.items()},
            "raw_output": self.raw_output,
        }


@dataclass
class DecompositionBenchmarkReport:
    domain: str
    locale: str
    split: str
    case_count: int
    average_score: float
    version: str = "1.0.0"
    cases: List[DecompositionCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "locale": self.locale,
            "split": self.split,
            "case_count": self.case_count,
            "average_score": round(self.average_score, 4),
            "cases": [item.to_dict() for item in self.cases],
        }


def _format_raw_output(parsed: Dict[str, Any]) -> str:
    lines = ["# 目标", parsed.get("totalGoal", "")]
    lines.append("# 任务拆解")
    for step in parsed.get("subSteps") or []:
        if step and step.upper() != "NULL":
            lines.append(f"- {step}")
    return "\n".join(lines)


async def evaluate_case(
    planner: TaskPlanner,
    case: DecompositionBenchmarkCase,
    *,
    registry: Any,
    lang: str = "zh",
) -> DecompositionCaseResult:
    parsed = await planner.run_decomposition(case.query, case.pre_survey, [])
    sub_steps = [step for step in parsed.get("subSteps") or [] if step and step.upper() != "NULL"]
    execution_order, depends_map = await planner.run_dependency_analysis(sub_steps)
    routed_subtasks = await planner.route_to_agents(sub_steps, execution_order, depends_map)
    routed_agents = [str(item.get("agent") or "").strip() for item in routed_subtasks if item.get("agent")]
    score = score_decomposition(
        parsed,
        case.expect,
        routed_subtasks=routed_subtasks,
        routed_agents=routed_agents,
        execution_order=execution_order,
        depends_map=depends_map,
        expect_routing=case.expect_routing,
        expect_dependency=case.expect_dependency,
        lang=lang,
    )
    return DecompositionCaseResult(
        case_id=case.case_id,
        query=case.query,
        score=score,
        total_goal=str(parsed.get("totalGoal") or ""),
        sub_steps=sub_steps,
        routed_agents=routed_agents,
        execution_order=list(execution_order),
        depends_map={key: list(value) for key, value in depends_map.items()},
        raw_output=_format_raw_output(parsed),
    )


async def evaluate_decomposition_benchmark(
    planner: TaskPlanner,
    *,
    registry: Any,
    fixtures: Optional[DecompositionFixtures] = None,
    split: str = "dev",
    lang: Optional[str] = None,
) -> DecompositionBenchmarkReport:
    loaded = fixtures or load_decomposition_fixtures()
    selected = loaded.cases_for_split(split)
    effective_lang = lang or loaded.locale

    case_results: List[DecompositionCaseResult] = []
    for case in selected:
        case_results.append(
            await evaluate_case(
                planner,
                case,
                registry=registry,
                lang=effective_lang,
            )
        )

    average_score = (
        sum(item.score.total for item in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    return DecompositionBenchmarkReport(
        domain=loaded.domain,
        locale=loaded.locale,
        split=split,
        case_count=len(case_results),
        average_score=average_score,
        version=loaded.version,
        cases=case_results,
    )
