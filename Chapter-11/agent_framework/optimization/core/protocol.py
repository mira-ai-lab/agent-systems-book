"""Optimizer protocol for Chapter-11 evolution SDK."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .result import OptimizationResult


@runtime_checkable
class DecompositionPromptOptimizer(Protocol):
    """Optimize travel ``decomposition_prompt`` against benchmark fixtures."""

    async def optimize(
        self,
        *,
        decomposition_prompt: str,
        registry: Any,
        executor_llm: Any,
        optimizer_llm: Any,
        fixtures: Any = None,
        max_steps: int = 10,
        failure_threshold: float = 0.8,
        rollback: bool = True,
        train_split: str = "train",
        dev_split: str = "dev",
    ) -> OptimizationResult:
        ...
