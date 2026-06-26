"""E2E benchmark 失败样本收集（供 E2E graph 优化使用）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase

from .evaluator import E2eCaseResult, evaluate_e2e_case
from .runtime import E2eOrchestrator


@dataclass
class E2eTrainEvaluation:
    """Single-pass train split evaluation (avoids duplicate orchestrator runs)."""

    failures: List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]]
    train_scores: List[float]
    case_results: List[E2eCaseResult]


async def evaluate_e2e_train_cases(
    orchestrator: E2eOrchestrator,
    cases: List[DecompositionBenchmarkCase],
    *,
    failure_threshold: float,
    timeout_sec: float | None = None,
    max_failure_cases: Optional[int] = None,
) -> E2eTrainEvaluation:
    """Evaluate train cases once; return failures (lowest scores first, capped) and scores."""
    case_results: List[E2eCaseResult] = []
    failures: List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]] = []

    for case in cases:
        result = await evaluate_e2e_case(
            orchestrator,
            case,
            timeout_sec=timeout_sec,
        )
        case_results.append(result)
        if result.score.total < failure_threshold:
            failures.append((result, case))

    failures.sort(key=lambda item: item[0].score.total)
    if max_failure_cases is not None and max_failure_cases > 0:
        failures = failures[:max_failure_cases]

    train_scores = [item.score.total for item in case_results]
    return E2eTrainEvaluation(
        failures=failures,
        train_scores=train_scores,
        case_results=case_results,
    )


async def collect_e2e_failures(
    orchestrator: E2eOrchestrator,
    cases: List[DecompositionBenchmarkCase],
    *,
    failure_threshold: float,
    timeout_sec: float | None = None,
    max_failure_cases: Optional[int] = None,
) -> List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]]:
    """返回 E2E 得分低于阈值的 (result, case) 列表。"""
    report = await evaluate_e2e_train_cases(
        orchestrator,
        cases,
        failure_threshold=failure_threshold,
        timeout_sec=timeout_sec,
        max_failure_cases=max_failure_cases,
    )
    return report.failures
