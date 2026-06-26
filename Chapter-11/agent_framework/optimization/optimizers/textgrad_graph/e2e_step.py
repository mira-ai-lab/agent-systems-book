"""E2E graph 单步 backward + TextualGradientDescent 更新。"""

from __future__ import annotations

from typing import List, Tuple

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase
from agent_framework.optimization.e2e.evaluator import E2eCaseResult

from .e2e_graph import PlannerPromptE2eGraph


def run_e2e_graph_step(
    graph: PlannerPromptE2eGraph,
    failures: List[Tuple[E2eCaseResult, DecompositionBenchmarkCase]],
    *,
    constraints: List[str],
) -> None:
    """对 E2E 失败 case 批量 forward，求和后 backward，更新 planner prompt Variable。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    tg, _, _, TextualGradientDescent = require_textgrad()

    if not failures:
        return

    losses = []
    for result, case in failures:
        _, loss = graph.forward_failure(result, case)
        losses.append(loss)

    total_loss = tg.sum(losses) if len(losses) > 1 else losses[0]
    total_loss.backward(graph._engine)

    params = graph.trainable_parameters()
    if not params:
        raise ValueError("E2E graph 中没有 requires_grad=True 的 prompt Variable")

    optimizer = TextualGradientDescent(
        parameters=params,
        engine=graph._engine,
        constraints=constraints,
    )
    optimizer.step()
    optimizer.zero_grad()
