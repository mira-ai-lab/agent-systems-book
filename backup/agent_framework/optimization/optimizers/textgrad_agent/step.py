"""单 Agent graph 单步 backward + TextualGradientDescent。"""

from __future__ import annotations

from typing import Callable, List, Optional

from agent_framework.optimization.agents.fixtures import SingleAgentCase

from .graph import SingleAgentTextGradGraph

TextGradForwardCallback = Callable[[SingleAgentCase], None]


def run_single_agent_graph_step(
    graph: SingleAgentTextGradGraph,
    failure_cases: List[SingleAgentCase],
    *,
    constraints: List[str],
    on_forward_case: Optional[TextGradForwardCallback] = None,
) -> None:
    """对失败 case 批量 forward，求和后 backward，更新 system_prompt Variable。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    tg, _, _, TextualGradientDescent = require_textgrad()

    if not failure_cases:
        return

    losses = []
    for case in failure_cases:
        if on_forward_case is not None:
            on_forward_case(case)
        _, loss = graph.forward_case(case)
        losses.append(loss)

    total_loss = tg.sum(losses) if len(losses) > 1 else losses[0]
    total_loss.backward(graph._engine)

    params = graph.trainable_parameters()
    optimizer = TextualGradientDescent(
        parameters=params,
        engine=graph._engine,
        constraints=constraints,
    )
    optimizer.step()
    optimizer.zero_grad()
