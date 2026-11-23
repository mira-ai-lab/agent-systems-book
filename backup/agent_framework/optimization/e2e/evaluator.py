"""Batch evaluator for travel end-to-end orchestration benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_framework.optimization.decomposition.fixtures import (
    DecompositionFixtures,
    load_decomposition_fixtures,
)

from .expectations import resolve_e2e_expect
from .runtime import E2eOrchestrator
from .scorer import E2eScore, score_e2e_run


@dataclass
class E2eCaseResult:
    case_id: str
    query: str
    score: E2eScore
    final_response: str = ""
    invoked_agents: List[str] = field(default_factory=list)
    completed_subtasks: int = 0
    trace_id: str = ""
    orchestration_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "score": self.score.to_dict(),
            "final_response": self.final_response,
            "invoked_agents": list(self.invoked_agents),
            "completed_subtasks": self.completed_subtasks,
            "trace_id": self.trace_id,
            "orchestration_mode": self.orchestration_mode,
        }


@dataclass
class E2eBenchmarkReport:
    domain: str
    locale: str
    split: str
    profile: str
    case_count: int
    average_score: float
    cases: List[E2eCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "locale": self.locale,
            "split": self.split,
            "profile": self.profile,
            "case_count": self.case_count,
            "average_score": round(self.average_score, 4),
            "cases": [item.to_dict() for item in self.cases],
        }


async def evaluate_e2e_case(
    orchestrator: E2eOrchestrator,
    case,
    *,
    timeout_sec: Optional[float] = None,
) -> E2eCaseResult:
    result = await orchestrator.process_request(
        case.query,
        thread_id=case.case_id,
        timeout_sec=timeout_sec,
    )
    expect = resolve_e2e_expect(case)
    score = score_e2e_run(result, expect)
    return E2eCaseResult(
        case_id=case.case_id,
        query=case.query,
        score=score,
        final_response=str(result.get("final_response") or ""),
        invoked_agents=score.invoked_agents,
        completed_subtasks=score.completed_subtasks,
        trace_id=str(result.get("trace_id") or ""),
        orchestration_mode=str(result.get("orchestration_mode") or ""),
    )


async def evaluate_e2e_benchmark(
    orchestrator: E2eOrchestrator,
    *,
    fixtures: Optional[DecompositionFixtures] = None,
    split: str = "dev",
    profile: str = "workflow",
    timeout_sec: Optional[float] = None,
) -> E2eBenchmarkReport:
    loaded = fixtures or load_decomposition_fixtures()
    selected = loaded.cases_for_split(split)

    case_results: List[E2eCaseResult] = []
    for case in selected:
        case_results.append(
            await evaluate_e2e_case(
                orchestrator,
                case,
                timeout_sec=timeout_sec,
            )
        )

    average_score = (
        sum(item.score.total for item in case_results) / len(case_results)
        if case_results
        else 0.0
    )
    return E2eBenchmarkReport(
        domain=loaded.domain,
        locale=loaded.locale,
        split=split,
        profile=profile,
        case_count=len(case_results),
        average_score=average_score,
        cases=case_results,
    )
