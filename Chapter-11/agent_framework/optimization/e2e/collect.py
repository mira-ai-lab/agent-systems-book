"""E2E benchmark 失败样本收集（供 B2 graph 优化使用）。"""

from __future__ import annotations

from typing import List, Tuple

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase

from .evaluator import E2eCaseResult, evaluate_e2e_case
from .runtime import E2eOrchestrator


async def collect_e2e_failures(
    orchestrator: E2eOrchestrator,
    cases: List[DecompositionBenchmarkCase],
    *,
    failure_threshold: float,
    timeout_sec: float | None = None,
) -> List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]]:
    """返回 E2E 得分低于阈值的 (result, case) 列表。"""
    failures: List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]] = []
    for case in cases:
        result = await evaluate_e2e_case(
            orchestrator,
            case,
            timeout_sec=timeout_sec,
        )
        if result.score.total < failure_threshold:
            failures.append((result, case))
    return failures
