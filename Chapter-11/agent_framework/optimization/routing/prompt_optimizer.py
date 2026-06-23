"""TextGrad-style optimization loop for travel agent_routing prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.optimization.decomposition.fixtures import DecompositionFixtures, load_decomposition_fixtures
from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.planner_runtime import build_routing_planner

from .evaluator import RoutingCaseResult, evaluate_routing_benchmark, evaluate_routing_case
from .optimizer_prompts import ROUTING_FAILURE_CASE_TEMPLATE, ROUTING_REVISION_TEMPLATE

REQUIRED_PLACEHOLDERS = ("{agent_team}", "{subtasks_json}")
LOCAL_OPTIMIZER_NAME = "local_prompt"


def extract_agent_routing_prompt(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("optimizer 返回空 agent_routing prompt")

    fenced = re.search(r"```(?:markdown|text|prompt)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    missing = [token for token in REQUIRED_PLACEHOLDERS if token not in text]
    if missing:
        raise ValueError(f"optimized agent_routing 缺少占位符: {missing}")
    return text


def format_routing_failure_feedback(
    failures: List[tuple[RoutingCaseResult, Any]],
) -> str:
    blocks: List[str] = []
    for result, case in failures:
        details = "; ".join(result.score.details) if result.score.details else "得分低于阈值"
        expected = ", ".join(
            f"{item.task_id}->{item.expected_agent}" for item in case.expect_routing.assignments
        )
        subtasks_input = json.dumps(
            {
                "sub_steps": case.routing_input.sub_steps,
                "execution_order": case.routing_input.execution_order,
                "depends_map": case.routing_input.depends_map,
            },
            ensure_ascii=False,
        )
        blocks.append(
            ROUTING_FAILURE_CASE_TEMPLATE.format(
                case_id=result.case_id,
                query=result.query,
                subtasks_input=subtasks_input,
                raw_output=result.raw_output or "(empty)",
                score=result.score.total,
                details=details,
                expected_assignments=expected,
            )
        )
    return "\n".join(blocks)


async def collect_routing_failures(
    planner,
    cases: List[Any],
    *,
    failure_threshold: float,
) -> List[tuple[RoutingCaseResult, Any]]:
    failures: List[tuple[RoutingCaseResult, Any]] = []
    for case in cases:
        result = await evaluate_routing_case(planner, case)
        if result.score.total < failure_threshold:
            failures.append((result, case))
    return failures


async def propose_routing_prompt_revision(
    optimizer_llm: ChatOpenAI,
    *,
    current_prompt: str,
    failures: List[tuple[RoutingCaseResult, Any]],
    agent_team: str,
) -> str:
    if not failures:
        return current_prompt

    prompt = ROUTING_REVISION_TEMPLATE.format(
        current_prompt=current_prompt,
        agent_team=agent_team,
        failure_feedback=format_routing_failure_feedback(failures),
    )
    response = await optimizer_llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return extract_agent_routing_prompt(content)


async def optimize_agent_routing_prompt(
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
    loaded = fixtures or load_decomposition_fixtures()
    train_cases = loaded.routing_cases_for_split(train_split)
    agent_team = registry.get_all_agents_text()

    best_prompt = agent_routing
    best_planner = build_routing_planner(
        best_prompt,
        executor_llm,
        registry,
        locale=loaded.locale,
        decomposition_prompt=decomposition_prompt,
    )

    baseline_report = await evaluate_routing_benchmark(
        best_planner,
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
                    optimizer=LOCAL_OPTIMIZER_NAME,
                )
            )
            break

        candidate_prompt = await propose_routing_prompt_revision(
            optimizer_llm,
            current_prompt=best_prompt,
            failures=failures,
            agent_team=agent_team,
        )
        candidate_planner = build_routing_planner(
            candidate_prompt,
            executor_llm,
            registry,
            locale=loaded.locale,
            decomposition_prompt=decomposition_prompt,
        )
        candidate_report = await evaluate_routing_benchmark(
            candidate_planner,
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
                optimizer=LOCAL_OPTIMIZER_NAME,
            )
        )

        if accepted and candidate_dev >= 0.999:
            break

    return OptimizationResult(
        best_prompt=best_prompt,
        baseline_dev_score=baseline_report.average_score,
        best_dev_score=best_dev_score,
        steps=steps,
        optimizer=LOCAL_OPTIMIZER_NAME,
    )
