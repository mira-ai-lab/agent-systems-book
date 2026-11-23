"""Planner graph 单步 backward + TextualGradientDescent 更新。"""

from __future__ import annotations

from typing import List

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase

from .graph import OptimizeSlot, PlannerTextGradGraph


def run_planner_graph_step(
    graph: PlannerTextGradGraph,
    failure_cases: List[DecompositionBenchmarkCase],
    *,
    constraints: List[str],
) -> None:
    """对失败 case 批量 forward，求和后 backward，更新可训练 prompt Variable。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    tg, _, _, TextualGradientDescent = require_textgrad()

    if not failure_cases:
        return

    losses = []
    for case in failure_cases:
        _, loss = graph.forward_case(case)
        losses.append(loss)

    total_loss = tg.sum(losses) if len(losses) > 1 else losses[0]
    total_loss.backward(graph._engine)

    params = graph.trainable_parameters()
    if not params:
        raise ValueError("graph 中没有 requires_grad=True 的 prompt Variable")

    optimizer = TextualGradientDescent(
        parameters=params,
        engine=graph._engine,
        constraints=constraints,
    )
    optimizer.step()
    optimizer.zero_grad()
