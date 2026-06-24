"""优化目标：L1/L2 planner 分 vs 端到端 E2E 分。"""

from __future__ import annotations

from typing import Literal

OptimizationObjective = Literal["l1_l2", "e2e"]
VALID_OBJECTIVES = ("l1_l2", "e2e")


def parse_optimization_objective(raw: str) -> OptimizationObjective:
    normalized = (raw or "l1_l2").strip().lower()
    if normalized not in VALID_OBJECTIVES:
        raise ValueError(f"不支持的 objective='{raw}'，可选: {', '.join(VALID_OBJECTIVES)}")
    return normalized  # type: ignore[return-value]
