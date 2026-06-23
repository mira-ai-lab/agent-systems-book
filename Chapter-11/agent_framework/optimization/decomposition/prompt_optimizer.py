"""TextGrad-style optimization loop for travel decomposition prompts (local optimizer)."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.planner_runtime import build_decomposition_planner

from .evaluator import DecompositionCaseResult, evaluate_case, evaluate_decomposition_benchmark
from .fixtures import DecompositionFixtures, load_decomposition_fixtures
from .optimizer_prompts import FAILURE_CASE_TEMPLATE, PROMPT_REVISION_TEMPLATE

REQUIRED_PLACEHOLDERS = ("{background_info}", "{agent_team}", "{user_input}")
LOCAL_OPTIMIZER_NAME = "local_prompt"


def extract_decomposition_prompt(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("optimizer 返回空 prompt")

    fenced = re.search(r"```(?:markdown|text|prompt)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    missing = [token for token in REQUIRED_PLACEHOLDERS if token not in text]
    if missing:
        raise ValueError(f"optimized prompt 缺少占位符: {missing}")
    return text


def format_failure_feedback_from_cases(
    failures: List[tuple[DecompositionCaseResult, Any]],
) -> str:
    blocks: List[str] = []
    for result, case in failures:
        details = "; ".join(result.score.details) if result.score.details else "得分低于阈值"
        blocks.append(
            FAILURE_CASE_TEMPLATE.format(
                case_id=result.case_id,
                query=result.query,
                raw_output=result.raw_output or "(empty)",
                score=result.score.total,
                details=details,
                expected_agents=", ".join(case.expect.mappable_agents) or "(none)",
                min_subtasks=case.expect.min_subtasks,
                max_subtasks=case.expect.max_subtasks,
            )
        )
    return "\n".join(blocks)


async def collect_failures(
    planner,
    cases: List[Any],
    *,
    registry: Any,
    lang: str,
    failure_threshold: float,
) -> List[tuple[DecompositionCaseResult, Any]]:
    failures: List[tuple[DecompositionCaseResult, Any]] = []
    for case in cases:
        result = await evaluate_case(planner, case, registry=registry, lang=lang)
        if result.score.total < failure_threshold:
            failures.append((result, case))
    return failures


async def propose_prompt_revision(
    optimizer_llm: ChatOpenAI,
    *,
    current_prompt: str,
    failures: List[tuple[DecompositionCaseResult, Any]],
    agent_team: str,
) -> str:
    if not failures:
        return current_prompt

    prompt = PROMPT_REVISION_TEMPLATE.format(
        current_prompt=current_prompt,
        agent_team=agent_team,
        failure_feedback=format_failure_feedback_from_cases(failures),
    )
    response = await optimizer_llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return extract_decomposition_prompt(content)


async def optimize_decomposition_prompt(
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
) -> OptimizationResult:
    loaded = fixtures or load_decomposition_fixtures()
    train_cases = loaded.cases_for_split(train_split)
    agent_team = registry.get_all_agents_text()

    best_prompt = decomposition_prompt
    baseline_report = await evaluate_decomposition_benchmark(
        build_decomposition_planner(best_prompt, executor_llm, registry, locale=loaded.locale),
        registry=registry,
        fixtures=loaded,
        split=dev_split,
    )
    best_dev_score = baseline_report.average_score
    steps: List[OptimizationStepRecord] = []

    for step in range(1, max_steps + 1):
        planner = build_decomposition_planner(best_prompt, executor_llm, registry, locale=loaded.locale)
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
                    optimizer=LOCAL_OPTIMIZER_NAME,
                )
            )
            break

        candidate_prompt = await propose_prompt_revision(
            optimizer_llm,
            current_prompt=best_prompt,
            failures=failures,
            agent_team=agent_team,
        )
        candidate_report = await evaluate_decomposition_benchmark(
            build_decomposition_planner(candidate_prompt, executor_llm, registry, locale=loaded.locale),
            registry=registry,
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
