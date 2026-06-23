"""Travel decomposition prompt ↔ textgrad.Variable adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent_framework.optimization.decomposition.prompt_optimizer import (
    REQUIRED_PLACEHOLDERS as DECOMPOSITION_REQUIRED_PLACEHOLDERS,
    extract_decomposition_prompt,
)
from agent_framework.optimization.routing.prompt_optimizer import (
    REQUIRED_PLACEHOLDERS as ROUTING_REQUIRED_PLACEHOLDERS,
    extract_agent_routing_prompt,
)

DECOMPOSITION_PROMPT_ROLE = "travel decomposition prompt template"
ROUTING_PROMPT_ROLE = "travel agent_routing prompt template"


def decomposition_prompt_variable(prompt: str):
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    return Variable(
        extract_decomposition_prompt(prompt),
        requires_grad=True,
        role_description=DECOMPOSITION_PROMPT_ROLE,
    )


def routing_prompt_variable(prompt: str):
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    return Variable(
        extract_agent_routing_prompt(prompt),
        requires_grad=True,
        role_description=ROUTING_PROMPT_ROLE,
    )


def failure_feedback_variable(feedback: str, *, role: str = "benchmark failure feedback"):
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    return Variable(feedback, requires_grad=False, role_description=role)


def read_decomposition_prompt_value(prompt_var) -> str:
    return extract_decomposition_prompt(str(getattr(prompt_var, "value", prompt_var)))


def read_routing_prompt_value(prompt_var) -> str:
    return extract_agent_routing_prompt(str(getattr(prompt_var, "value", prompt_var)))


def read_prompt_value(prompt_var) -> str:
    return read_decomposition_prompt_value(prompt_var)


def apply_decomposition_prompt_value(prompts: Any, prompt_value: str):
    cleaned = extract_decomposition_prompt(prompt_value)
    missing = [token for token in DECOMPOSITION_REQUIRED_PLACEHOLDERS if token not in cleaned]
    if missing:
        raise ValueError(f"optimized prompt 缺少占位符: {missing}")
    return replace(prompts, decomposition_prompt=cleaned)


def apply_routing_prompt_value(prompts: Any, prompt_value: str):
    cleaned = extract_agent_routing_prompt(prompt_value)
    missing = [token for token in ROUTING_REQUIRED_PLACEHOLDERS if token not in cleaned]
    if missing:
        raise ValueError(f"optimized agent_routing 缺少占位符: {missing}")
    return replace(prompts, agent_routing=cleaned)


def apply_prompt_value(prompts: Any, prompt_value: str):
    return apply_decomposition_prompt_value(prompts, prompt_value)
