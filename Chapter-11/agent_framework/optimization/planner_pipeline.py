"""Run travel planner prompt optimization across decomposition and routing slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Set

from langchain_openai import ChatOpenAI

from agent_framework.optimization.core.result import OptimizationResult
from agent_framework.optimization.decomposition.fixtures import DecompositionFixtures
from agent_framework.optimization.decomposition.prompt_optimizer import optimize_decomposition_prompt
from agent_framework.optimization.objective import OptimizationObjective
from agent_framework.optimization.optimizers.textgrad_lib.decomposition import (
    optimize_decomposition_prompt_textgrad,
)
from agent_framework.optimization.optimizers.textgrad_lib.routing import (
    optimize_agent_routing_prompt_textgrad,
)
from agent_framework.optimization.optimizers.textgrad_graph.decomposition import (
    optimize_decomposition_prompt_graph,
)
from agent_framework.optimization.optimizers.textgrad_graph.routing import (
    optimize_agent_routing_prompt_graph,
)
from agent_framework.optimization.routing.prompt_optimizer import optimize_agent_routing_prompt

OptimizerBackend = Literal["local", "textgrad_lib", "textgrad_graph"]
PlannerSlot = Literal["decomposition", "routing"]
VALID_SLOTS = ("decomposition", "routing")


@dataclass
class PlannerOptimizationOutput:
    backend: str
    slots: List[str]
    decomposition_result: Optional[OptimizationResult] = None
    routing_result: Optional[OptimizationResult] = None
    decomposition_prompt: Optional[str] = None
    agent_routing: Optional[str] = None


def parse_planner_slots(raw: str) -> List[PlannerSlot]:
    normalized = (raw or "all").strip().lower()
    if normalized == "all":
        return ["decomposition", "routing"]

    selected: List[PlannerSlot] = []
    for token in normalized.split(","):
        slot = token.strip().lower()
        if not slot:
            continue
        if slot not in VALID_SLOTS:
            raise ValueError(f"不支持的 slot='{slot}'，可选: {', '.join(VALID_SLOTS)}, all")
        if slot not in selected:
            selected.append(slot)  # type: ignore[arg-type]
    if not selected:
        raise ValueError("slots 不能为空")
    return selected


async def run_planner_optimization(
    *,
    backend: OptimizerBackend,
    slots: List[PlannerSlot],
    decomposition_prompt: str,
    agent_routing: str,
    registry: Any,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: DecompositionFixtures,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    objective: OptimizationObjective = "l1_l2",
    e2e_profile: str = "workflow",
    e2e_timeout_sec: Optional[float] = None,
    enable_guess_agent: bool = True,
    max_failure_cases_per_step: int = 3,
) -> PlannerOptimizationOutput:
    current_decomposition = decomposition_prompt
    current_routing = agent_routing
    decomposition_result: Optional[OptimizationResult] = None
    routing_result: Optional[OptimizationResult] = None
    slot_set: Set[str] = set(slots)

    if "decomposition" in slot_set:
        if backend == "textgrad_graph":
            decomposition_result = await optimize_decomposition_prompt_graph(
                decomposition_prompt=current_decomposition,
                agent_routing=current_routing,
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
                max_failure_cases_per_step=max_failure_cases_per_step,
            )
        elif backend == "textgrad_lib":
            decomposition_result = await optimize_decomposition_prompt_textgrad(
                decomposition_prompt=current_decomposition,
                registry=registry,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                fixtures=fixtures,
                max_steps=max_steps,
                failure_threshold=failure_threshold,
                rollback=rollback,
                train_split=train_split,
                dev_split=dev_split,
            )
        else:
            decomposition_result = await optimize_decomposition_prompt(
                decomposition_prompt=current_decomposition,
                registry=registry,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                fixtures=fixtures,
                max_steps=max_steps,
                failure_threshold=failure_threshold,
                rollback=rollback,
                train_split=train_split,
                dev_split=dev_split,
            )
        current_decomposition = decomposition_result.best_prompt

    if "routing" in slot_set:
        if backend == "textgrad_graph":
            routing_result = await optimize_agent_routing_prompt_graph(
                agent_routing=current_routing,
                registry=registry,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                fixtures=fixtures,
                max_steps=max_steps,
                failure_threshold=failure_threshold,
                rollback=rollback,
                train_split=train_split,
                dev_split=dev_split,
                decomposition_prompt=current_decomposition,
                objective=objective,
                e2e_profile=e2e_profile,
                e2e_timeout_sec=e2e_timeout_sec,
                enable_guess_agent=enable_guess_agent,
                max_failure_cases_per_step=max_failure_cases_per_step,
            )
        elif backend == "textgrad_lib":
            routing_result = await optimize_agent_routing_prompt_textgrad(
                agent_routing=current_routing,
                registry=registry,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                fixtures=fixtures,
                max_steps=max_steps,
                failure_threshold=failure_threshold,
                rollback=rollback,
                train_split=train_split,
                dev_split=dev_split,
                decomposition_prompt=current_decomposition,
            )
        else:
            routing_result = await optimize_agent_routing_prompt(
                agent_routing=current_routing,
                registry=registry,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                fixtures=fixtures,
                max_steps=max_steps,
                failure_threshold=failure_threshold,
                rollback=rollback,
                train_split=train_split,
                dev_split=dev_split,
                decomposition_prompt=current_decomposition,
            )
        current_routing = routing_result.best_prompt

    return PlannerOptimizationOutput(
        backend=backend,
        slots=list(slots),
        decomposition_result=decomposition_result,
        routing_result=routing_result,
        decomposition_prompt=current_decomposition,
        agent_routing=current_routing,
    )
