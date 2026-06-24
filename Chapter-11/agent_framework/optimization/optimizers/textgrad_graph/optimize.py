"""Planner graph 优化循环（decomposition / routing；支持 L1/L2 与 E2E 目标）。"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.decomposition.evaluator import evaluate_case, evaluate_decomposition_benchmark
from agent_framework.optimization.decomposition.fixtures import DecompositionFixtures, load_decomposition_fixtures
from agent_framework.optimization.decomposition.prompt_optimizer import collect_failures, extract_decomposition_prompt
from agent_framework.optimization.e2e.collect import collect_e2e_failures
from agent_framework.optimization.e2e.evaluator import evaluate_e2e_benchmark, evaluate_e2e_case
from agent_framework.optimization.e2e.runtime import build_e2e_orchestrator
from agent_framework.optimization.objective import OptimizationObjective
from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad
from agent_framework.optimization.planner_runtime import build_decomposition_planner, build_routing_planner
from agent_framework.optimization.routing.evaluator import evaluate_routing_benchmark
from agent_framework.optimization.routing.prompt_optimizer import extract_agent_routing_prompt

from .e2e_graph import TEXTGRAD_GRAPH_E2E_OPTIMIZER_NAME, PlannerPromptE2eGraph
from .e2e_step import run_e2e_graph_step
from .graph import OptimizeSlot, PlannerTextGradGraph
from .prompts import DECOMPOSITION_GRAPH_CONSTRAINTS, ROUTING_GRAPH_CONSTRAINTS
from .step import run_planner_graph_step

TEXTGRAD_GRAPH_OPTIMIZER_NAME = "textgrad_graph"


def _optimizer_name(objective: OptimizationObjective) -> str:
    return TEXTGRAD_GRAPH_E2E_OPTIMIZER_NAME if objective == "e2e" else TEXTGRAD_GRAPH_OPTIMIZER_NAME


def _build_e2e_orchestrator(
    *,
    executor_llm: ChatOpenAI,
    fixtures: DecompositionFixtures,
    decomposition_prompt: str,
    agent_routing: str,
    e2e_profile: str,
    enable_guess_agent: bool,
):
    return build_e2e_orchestrator(
        executor_llm,
        locale=fixtures.locale,
        profile=e2e_profile,
        enable_memory=False,
        enable_guess_agent=enable_guess_agent,
        prompt_overrides={
            "decomposition_prompt": extract_decomposition_prompt(decomposition_prompt),
            "agent_routing": extract_agent_routing_prompt(agent_routing),
        },
        use_optimized=False,
    )


async def _evaluate_dev_l1_l2(
    *,
    slot: OptimizeSlot,
    decomposition_prompt: str,
    agent_routing: str,
    executor_llm: ChatOpenAI,
    registry: Any,
    fixtures: DecompositionFixtures,
    dev_split: str,
) -> float:
    if slot == "routing":
        report = await evaluate_routing_benchmark(
            build_routing_planner(
                extract_agent_routing_prompt(agent_routing),
                executor_llm,
                registry,
                locale=fixtures.locale,
                decomposition_prompt=extract_decomposition_prompt(decomposition_prompt),
            ),
            fixtures=fixtures,
            split=dev_split,
        )
        return report.average_score

    report = await evaluate_decomposition_benchmark(
        build_decomposition_planner(
            extract_decomposition_prompt(decomposition_prompt),
            executor_llm,
            registry,
            locale=fixtures.locale,
            agent_routing=extract_agent_routing_prompt(agent_routing),
        ),
        registry=registry,
        fixtures=fixtures,
        split=dev_split,
    )
    return report.average_score


async def _evaluate_dev_e2e(
    *,
    executor_llm: ChatOpenAI,
    fixtures: DecompositionFixtures,
    decomposition_prompt: str,
    agent_routing: str,
    dev_split: str,
    e2e_profile: str,
    e2e_timeout_sec: Optional[float],
    enable_guess_agent: bool,
) -> float:
    orchestrator = _build_e2e_orchestrator(
        executor_llm=executor_llm,
        fixtures=fixtures,
        decomposition_prompt=decomposition_prompt,
        agent_routing=agent_routing,
        e2e_profile=e2e_profile,
        enable_guess_agent=enable_guess_agent,
    )
    report = await evaluate_e2e_benchmark(
        orchestrator,
        fixtures=fixtures,
        split=dev_split,
        profile=e2e_profile,
        timeout_sec=e2e_timeout_sec,
    )
    return report.average_score


async def _evaluate_dev_for_objective(
    *,
    objective: OptimizationObjective,
    slot: OptimizeSlot,
    decomposition_prompt: str,
    agent_routing: str,
    executor_llm: ChatOpenAI,
    registry: Any,
    fixtures: DecompositionFixtures,
    dev_split: str,
    e2e_profile: str,
    e2e_timeout_sec: Optional[float],
    enable_guess_agent: bool,
) -> float:
    if objective == "e2e":
        return await _evaluate_dev_e2e(
            executor_llm=executor_llm,
            fixtures=fixtures,
            decomposition_prompt=decomposition_prompt,
            agent_routing=agent_routing,
            dev_split=dev_split,
            e2e_profile=e2e_profile,
            e2e_timeout_sec=e2e_timeout_sec,
            enable_guess_agent=enable_guess_agent,
        )
    return await _evaluate_dev_l1_l2(
        slot=slot,
        decomposition_prompt=decomposition_prompt,
        agent_routing=agent_routing,
        executor_llm=executor_llm,
        registry=registry,
        fixtures=fixtures,
        dev_split=dev_split,
    )


async def optimize_planner_prompt_graph(
    *,
    slot: OptimizeSlot,
    decomposition_prompt: str,
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
    objective: OptimizationObjective = "l1_l2",
    e2e_profile: str = "workflow",
    e2e_timeout_sec: Optional[float] = None,
    enable_guess_agent: bool = True,
) -> OptimizationResult:
    """通过 textgrad 计算图优化 planner prompt。

    objective:
      - ``l1_l2``: B1，Planner 三步 graph + L1/L2 dev rollback
      - ``e2e``: B2，完整 E2E 编排 graph + E2E dev rollback
    """
    require_textgrad()
    loaded = fixtures or load_decomposition_fixtures()
    train_cases = loaded.cases_for_split(train_split)
    optimizer_label = _optimizer_name(objective)

    best_decomposition = extract_decomposition_prompt(decomposition_prompt)
    best_routing = extract_agent_routing_prompt(agent_routing)
    best_prompt = best_decomposition if slot == "decomposition" else best_routing
    constraints = (
        DECOMPOSITION_GRAPH_CONSTRAINTS if slot == "decomposition" else ROUTING_GRAPH_CONSTRAINTS
    )

    baseline_dev = await _evaluate_dev_for_objective(
        objective=objective,
        slot=slot,
        decomposition_prompt=best_decomposition,
        agent_routing=best_routing,
        executor_llm=executor_llm,
        registry=registry,
        fixtures=loaded,
        dev_split=dev_split,
        e2e_profile=e2e_profile,
        e2e_timeout_sec=e2e_timeout_sec,
        enable_guess_agent=enable_guess_agent,
    )
    best_dev_score = baseline_dev
    steps: List[OptimizationStepRecord] = []

    for step in range(1, max_steps + 1):
        if objective == "e2e":
            orchestrator = _build_e2e_orchestrator(
                executor_llm=executor_llm,
                fixtures=loaded,
                decomposition_prompt=best_decomposition,
                agent_routing=best_routing,
                e2e_profile=e2e_profile,
                enable_guess_agent=enable_guess_agent,
            )
            failures = await collect_e2e_failures(
                orchestrator,
                train_cases,
                failure_threshold=failure_threshold,
                timeout_sec=e2e_timeout_sec,
            )
            train_scores = []
            for case in train_cases:
                item = await evaluate_e2e_case(
                    orchestrator,
                    case,
                    timeout_sec=e2e_timeout_sec,
                )
                train_scores.append(item.score.total)
        else:
            planner = build_decomposition_planner(
                best_decomposition,
                executor_llm,
                registry,
                locale=loaded.locale,
                agent_routing=best_routing,
            )
            failures = await collect_failures(
                planner,
                train_cases,
                registry=registry,
                lang=loaded.locale,
                failure_threshold=failure_threshold,
            )
            train_scores = []
            for case in train_cases:
                result = await evaluate_case(planner, case, registry=registry, lang=loaded.locale)
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
                    optimizer=optimizer_label,
                )
            )
            break

        if objective == "e2e":
            graph = PlannerPromptE2eGraph.create(
                executor_llm=executor_llm,
                locale=loaded.locale,
                decomposition_prompt=best_decomposition,
                agent_routing=best_routing,
                optimize_slot=slot,
                optimizer_llm=optimizer_llm,
                e2e_profile=e2e_profile,
                e2e_timeout_sec=e2e_timeout_sec,
                enable_guess_agent=enable_guess_agent,
            )
            failure_cases = [case for _, case in failures]
            run_e2e_graph_step(graph, failure_cases, constraints=constraints)
            candidate_decomposition, candidate_routing = graph.read_optimized_prompts()
        else:
            graph = PlannerTextGradGraph.create(
                executor_llm=executor_llm,
                registry=registry,
                locale=loaded.locale,
                decomposition_prompt=best_decomposition,
                agent_routing=best_routing,
                optimize_slot=slot,
                optimizer_llm=optimizer_llm,
            )
            failure_cases = [case for _, case in failures]
            run_planner_graph_step(graph, failure_cases, constraints=constraints)
            candidate_decomposition, candidate_routing = graph.read_optimized_prompts()

        candidate_prompt = candidate_decomposition if slot == "decomposition" else candidate_routing

        candidate_dev = await _evaluate_dev_for_objective(
            objective=objective,
            slot=slot,
            decomposition_prompt=candidate_decomposition,
            agent_routing=candidate_routing,
            executor_llm=executor_llm,
            registry=registry,
            fixtures=loaded,
            dev_split=dev_split,
            e2e_profile=e2e_profile,
            e2e_timeout_sec=e2e_timeout_sec,
            enable_guess_agent=enable_guess_agent,
        )
        accepted = should_accept_candidate(candidate_dev, best_dev_score, rollback=rollback)

        if accepted:
            best_decomposition = candidate_decomposition
            best_routing = candidate_routing
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
                optimizer=optimizer_label,
            )
        )

        if accepted and candidate_dev >= 0.999:
            break

    return OptimizationResult(
        best_prompt=best_prompt,
        baseline_dev_score=baseline_dev,
        best_dev_score=best_dev_score,
        steps=steps,
        optimizer=optimizer_label,
    )
