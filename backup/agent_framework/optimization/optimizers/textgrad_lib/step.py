"""Shared textgrad TextualGradientDescent step for prompt variables."""

from __future__ import annotations

from typing import List

from ._import import require_textgrad


def run_textgrad_prompt_step(
    prompt_var,
    engine,
    failure_feedback: str,
    *,
    loss_prompt: str,
    constraints: List[str],
) -> None:
    _, Variable, TextLoss, TextualGradientDescent = require_textgrad()

    loss_instruction = Variable(
        f"{loss_prompt}\n\n# Failure cases\n{failure_feedback}",
        requires_grad=False,
        role_description="failure feedback for prompt optimization",
    )
    loss_fn = TextLoss(loss_instruction, engine)
    optimizer = TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,
        constraints=constraints,
    )

    critique_context = Variable(
        failure_feedback,
        requires_grad=False,
        role_description="benchmark failure details",
    )
    combined = Variable(
        f"Prompt template:\n{prompt_var.value}\n\nFailures:\n{critique_context.value}",
        requires_grad=False,
        role_description="prompt plus failures",
    )
    loss = loss_fn(combined)
    loss.backward(engine)
    optimizer.step()
    optimizer.zero_grad()
