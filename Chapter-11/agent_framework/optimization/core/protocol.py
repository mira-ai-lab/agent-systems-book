"""Chapter-11 进化 SDK 的 Optimizer 协议定义。

通过 ``Protocol`` 约定各 Optimizer 的对外接口，便于 local / textgrad_lib / 未来 MIPRO 等实现互换。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .result import OptimizationResult


@runtime_checkable
class DecompositionPromptOptimizer(Protocol):
    """旅行域 ``decomposition_prompt`` 优化器协议。

    实现方需根据 benchmark fixtures 迭代优化任务拆解 prompt，
    并返回包含 best_prompt 与逐步记录的 ``OptimizationResult``。
    """

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
