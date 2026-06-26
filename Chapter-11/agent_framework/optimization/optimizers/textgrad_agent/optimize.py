"""任意旅行子 Agent system_prompt 的 textgrad graph 优化（Agent-B1/B2 通用入口）。"""

from __future__ import annotations

from typing import Callable, List, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agent_prompt_store import optimized_agent_prompts_path
from agent_framework.optimization.agents.collect import collect_single_agent_failures
from agent_framework.optimization.agents.evaluator import (
    CaseEvalProgressCallback,
    create_agent_bridge,
    evaluate_single_agent_benchmark,
    evaluate_single_agent_case,
)
from agent_framework.optimization.agents.fixtures import SingleAgentCase
from agent_framework.optimization.agents.fixtures import SingleAgentCaseFixtures, load_single_agent_cases
from agent_framework.optimization.agents.runtime import (
    TRAVEL_OPTIMIZABLE_AGENTS,
    default_agent_prompt_template,
)
from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

from .graph import TEXTGRAD_AGENT_GRAPH_OPTIMIZER_NAME, SingleAgentTextGradGraph
from .loss import agent_graph_constraints
from .step import run_single_agent_graph_step

FLIGHT_AGENT_NAME = "FlightAgent"


async def optimize_agent_prompt_graph(
    *,
    agent_name: str,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: Optional[SingleAgentCaseFixtures] = None,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    system_prompt_template: Optional[str] = None,
    on_case_evaluated: Optional[CaseEvalProgressCallback] = None,
) -> OptimizationResult:
    """优化指定子 Agent 的 system_prompt（单节点 graph，生产 LangGraph 不改动）。"""
    if agent_name not in TRAVEL_OPTIMIZABLE_AGENTS:
        raise ValueError(f"不支持的 agent_name={agent_name!r}")

    require_textgrad()
    loaded = fixtures or load_single_agent_cases()
    train_cases = loaded.cases_for_split(train_split, agent_name=agent_name)

    best_template = system_prompt_template or default_agent_prompt_template(
        agent_name, locale=loaded.locale
    )
    bridge = create_agent_bridge(executor_llm, agent_name=agent_name, locale=loaded.locale)
    constraints = agent_graph_constraints(agent_name)

    baseline_report = await evaluate_single_agent_benchmark(
        bridge,
        fixtures=loaded,
        agent_name=agent_name,
        split=dev_split,
        system_prompt_template=best_template,
        phase="baseline_dev",
        on_case_evaluated=on_case_evaluated,
    )
    best_dev_score = baseline_report.average_score
    steps: List[OptimizationStepRecord] = []

    for step in range(1, max_steps + 1):
        failures = await collect_single_agent_failures(
            bridge,
            train_cases,
            system_prompt_template=best_template,
            failure_threshold=failure_threshold,
            on_case_evaluated=on_case_evaluated,
        )

        train_scores = []
        for case in train_cases:
            item = await evaluate_single_agent_case(
                bridge,
                case,
                system_prompt_template=best_template,
                phase="train_score",
                on_case_evaluated=on_case_evaluated,
            )
            train_scores.append(item.score.total)
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
                    prompt_preview=best_template[:160],
                    optimizer=TEXTGRAD_AGENT_GRAPH_OPTIMIZER_NAME,
                )
            )
            break

        graph = SingleAgentTextGradGraph.create(
            executor_llm=executor_llm,
            locale=loaded.locale,
            system_prompt_template=best_template,
            agent_name=agent_name,
            optimizer_llm=optimizer_llm,
        )
        failure_cases = [case for _, case in failures]
        on_forward_case: Optional[Callable[[SingleAgentCase], None]] = None
        if on_case_evaluated is not None:

            def on_forward_case(case: SingleAgentCase) -> None:
                print(
                    f"[{agent_name}] textgrad_forward {case.case_id} (step={step})",
                    flush=True,
                )

        run_single_agent_graph_step(
            graph,
            failure_cases,
            constraints=constraints,
            on_forward_case=on_forward_case,
        )
        if on_case_evaluated is not None:
            print(f"[{agent_name}] textgrad_step step={step} updating prompt", flush=True)
        candidate_template = graph.read_optimized_prompt_template()

        candidate_report = await evaluate_single_agent_benchmark(
            bridge,
            fixtures=loaded,
            agent_name=agent_name,
            split=dev_split,
            system_prompt_template=candidate_template,
            phase="candidate_dev",
            on_case_evaluated=on_case_evaluated,
        )
        candidate_dev = candidate_report.average_score
        accepted = should_accept_candidate(candidate_dev, best_dev_score, rollback=rollback)

        if accepted:
            best_template = candidate_template
            best_dev_score = candidate_dev

        steps.append(
            OptimizationStepRecord(
                step=step,
                train_average=train_average,
                dev_average=best_dev_score,
                candidate_dev_average=candidate_dev,
                accepted=accepted,
                failure_count=len(failures),
                prompt_preview=candidate_template[:160],
                optimizer=TEXTGRAD_AGENT_GRAPH_OPTIMIZER_NAME,
            )
        )

        if accepted and candidate_dev >= 0.999:
            break

    return OptimizationResult(
        best_prompt=best_template,
        baseline_dev_score=baseline_report.average_score,
        best_dev_score=best_dev_score,
        steps=steps,
        optimizer=TEXTGRAD_AGENT_GRAPH_OPTIMIZER_NAME,
    )


async def optimize_flight_agent_prompt_graph(
    *,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: Optional[SingleAgentCaseFixtures] = None,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    system_prompt_template: Optional[str] = None,
    on_case_evaluated: Optional[CaseEvalProgressCallback] = None,
) -> OptimizationResult:
    """B1 兼容：优化 FlightAgent system_prompt。"""
    return await optimize_agent_prompt_graph(
        agent_name=FLIGHT_AGENT_NAME,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=max_steps,
        failure_threshold=failure_threshold,
        rollback=rollback,
        train_split=train_split,
        dev_split=dev_split,
        system_prompt_template=system_prompt_template,
        on_case_evaluated=on_case_evaluated,
    )


def default_agent_report_path(agent_name: str, locale: str = "zh") -> str:
    """默认单 Agent 优化报告 JSON 路径。"""
    slug = agent_name.replace("Agent", "").lower()
    return str(
        optimized_agent_prompts_path(locale).parent / f"{slug}_agent_textgrad_graph_report.json"
    )


def default_flight_agent_report_path(locale: str = "zh") -> str:
    """B1 兼容：FlightAgent 报告路径。"""
    return default_agent_report_path(FLIGHT_AGENT_NAME, locale)
