"""Agent routing prompt optimizer using the ``textgrad`` library."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.decomposition.fixtures import DecompositionFixtures, load_decomposition_fixtures
from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad
from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
    read_routing_prompt_value,
    routing_prompt_variable,
)
from agent_framework.optimization.optimizers.textgrad_lib.engine import create_textgrad_engine
from agent_framework.optimization.optimizers.textgrad_lib.prompts import (
    ROUTING_TEXTGRAD_CONSTRAINTS,
    ROUTING_TEXTGRAD_LOSS_PROMPT,
)
from agent_framework.optimization.optimizers.textgrad_lib.step import run_textgrad_prompt_step
from agent_framework.optimization.planner_runtime import build_routing_planner
from agent_framework.optimization.routing.evaluator import evaluate_routing_benchmark, evaluate_routing_case
from agent_framework.optimization.routing.prompt_optimizer import (
    collect_routing_failures,
    extract_agent_routing_prompt,
    format_routing_failure_feedback,
)

TEXTGRAD_LIB_OPTIMIZER_NAME = "textgrad_lib"


async def optimize_agent_routing_prompt_textgrad(
    *,
    agent_routing: str,
    registry: Any,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: Optional[DecompositionFixtures] = None,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    decomposition_prompt: Optional[str] = None,
) -> OptimizationResult:
    """Optimize agent_routing prompt with ``textgrad`` TextualGradientDescent."""
    require_textgrad()
    loaded = fixtures or load_decomposition_fixtures()
    train_cases = loaded.routing_cases_for_split(train_split)
    engine = create_textgrad_engine(optimizer_llm)

    best_prompt = extract_agent_routing_prompt(agent_routing)
    baseline_report = await evaluate_routing_benchmark(
        build_routing_planner(
            best_prompt,
            executor_llm,
            registry,
            locale=loaded.locale,
            decomposition_prompt=decomposition_prompt,
        ),
        fixtures=loaded,
        split=dev_split,
    )
    best_dev_score = baseline_report.average_score
    steps: List[OptimizationStepRecord] = []

    for step in range(1, max_steps + 1):
        planner = build_routing_planner(
            best_prompt,
            executor_llm,
            registry,
            locale=loaded.locale,
            decomposition_prompt=decomposition_prompt,
        )
        failures = await collect_routing_failures(
            planner,
            train_cases,
            failure_threshold=failure_threshold,
        )

        train_scores = []
        for case in train_cases:
            result = await evaluate_routing_case(planner, case)
            train_scores.append(result.score.total)
        train_average = sum(train_scores) / len(train_scores) if train_scores else 0.0

        if not failures:
            steps.append(
                OptimizationStepRecord(
                    step=step,
                    train_average=train_average,
                    dev_average=best_dev_score,
                    candidate_dev_average=best_dev_score,
                    accepted=False,
                    failure_count=0,
                    prompt_preview=best_prompt[:160],
                    optimizer=TEXTGRAD_LIB_OPTIMIZER_NAME,
                )
            )
            break

        prompt_var = routing_prompt_variable(best_prompt)
        run_textgrad_prompt_step(
            prompt_var,
            engine,
            format_routing_failure_feedback(failures),
            loss_prompt=ROUTING_TEXTGRAD_LOSS_PROMPT,
            constraints=ROUTING_TEXTGRAD_CONSTRAINTS,
        )
        candidate_prompt = read_routing_prompt_value(prompt_var)

        candidate_report = await evaluate_routing_benchmark(
            build_routing_planner(
                candidate_prompt,
                executor_llm,
                registry,
                locale=loaded.locale,
                decomposition_prompt=decomposition_prompt,
            ),
            fixtures=loaded,
            split=dev_split,
        )
        candidate_dev = candidate_report.average_score
        accepted = should_accept_candidate(candidate_dev, best_dev_score, rollback=rollback)

        if accepted:
            best_prompt = candidate_prompt
            best_dev_score = candidate_dev

        steps.append(
            OptimizationStepRecord(
                step=step,
                train_average=train_average,
                dev_average=best_dev_score,
                candidate_dev_average=candidate_dev,
                accepted=accepted,
                failure_count=len(failures),
                prompt_preview=candidate_prompt[:160],
                optimizer=TEXTGRAD_LIB_OPTIMIZER_NAME,
            )
        )

        if accepted and candidate_dev >= 0.999:
            break

    return OptimizationResult(
        best_prompt=best_prompt,
        baseline_dev_score=baseline_report.average_score,
        best_dev_score=best_dev_score,
        steps=steps,
        optimizer=TEXTGRAD_LIB_OPTIMIZER_NAME,
    )
