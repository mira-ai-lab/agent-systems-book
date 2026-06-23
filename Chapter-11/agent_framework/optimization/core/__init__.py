"""Shared optimization primitives (rollback, results, save helpers)."""

from .result import OptimizationResult, OptimizationStepRecord
from .rollback import should_accept_candidate
from .save import (
    save_decomposition_optimization_artifacts,
    save_planner_optimization_artifacts,
    save_routing_optimization_artifacts,
)

__all__ = [
    "OptimizationResult",
    "OptimizationStepRecord",
    "should_accept_candidate",
    "save_decomposition_optimization_artifacts",
    "save_routing_optimization_artifacts",
    "save_planner_optimization_artifacts",
]
