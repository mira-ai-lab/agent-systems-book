"""Decomposition prompt optimizer via TaskPlanner textgrad graph (Phase B1)."""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.core.result import OptimizationResult
from agent_framework.optimization.decomposition.fixtures import DecompositionFixtures
from agent_framework.optimization.objective import OptimizationObjective

from .optimize import optimize_planner_prompt_graph


async def optimize_decomposition_prompt_graph(
    *,
    decomposition_prompt: str,
    registry: Any,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: Optional[DecompositionFixtures] = None,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    agent_routing: Optional[str] = None,
    objective: OptimizationObjective = "l1_l2",
    e2e_profile: str = "workflow",
    e2e_timeout_sec: Optional[float] = None,
    enable_guess_agent: bool = True,
) -> OptimizationResult:
    """优化 decomposition_prompt；routing prompt 固定（可传入当前 agent_routing）。"""
    from domains.travel.prompt_bundle import TravelPrompts

    loaded = fixtures
    locale = loaded.locale if loaded is not None else "zh"
    routing = agent_routing or TravelPrompts.build(locale=locale, use_optimized=False).agent_routing
    return await optimize_planner_prompt_graph(
        slot="decomposition",
        decomposition_prompt=decomposition_prompt,
        agent_routing=routing,
        registry=registry,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=max_steps,
        failure_threshold=failure_threshold,
        rollback=rollback,
        train_split=train_split,
        dev_split=dev_split,
        objective=objective,
        e2e_profile=e2e_profile,
        e2e_timeout_sec=e2e_timeout_sec,
        enable_guess_agent=enable_guess_agent,
    )
