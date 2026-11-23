"""Agent-B3：以 mini-pipeline dev 分为 rollback 目标的子 Agent prompt 优化。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agents.mini_pipeline.collect import (
    collect_agent_step_failures,
    collect_mini_pipeline_failures,
)
from agent_framework.optimization.agents.mini_pipeline.evaluator import (
    evaluate_mini_pipeline_benchmark,
    evaluate_mini_pipeline_case,
)
from agent_framework.optimization.agents.mini_pipeline.fixtures import (
    MiniPipelineFixtures,
    load_mini_pipeline_cases,
)
from agent_framework.optimization.agents.mini_pipeline.runtime import MiniPipelineRunner
from agent_framework.optimization.agents.runtime import TRAVEL_OPTIMIZABLE_AGENTS
from agent_framework.optimization.core.result import OptimizationResult, OptimizationStepRecord
from agent_framework.optimization.core.rollback import should_accept_candidate
from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

from .graph import SingleAgentTextGradGraph
from .loss import agent_graph_constraints
from .step import run_single_agent_graph_step

TEXTGRAD_AGENT_MINI_PIPELINE_OPTIMIZER_NAME = "textgrad_agent_mini_pipeline"


async def optimize_agent_prompt_mini_pipeline(
    *,
    agent_name: str,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    prompt_templates: Mapping[str, str],
    fixtures: Optional[MiniPipelineFixtures] = None,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    step_failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    system_prompt_template: Optional[str] = None,
) -> OptimizationResult:
    """Agent-B3：优化单 Agent prompt，rollback 以 mini-pipeline dev 分为准。

    与 B1/B2 的区别：
    - 评测 / rollback 走固定 subtask 串联的 mini-pipeline，而非单 Agent benchmark
    - textgrad 反传仍用单节点 graph（只更新当前 slot 的 system_prompt）
    - ``prompt_templates`` 提供其它 Agent 的当前 prompt（前面 slot 已优化结果会传入）
    """
    if agent_name not in TRAVEL_OPTIMIZABLE_AGENTS:
        raise ValueError(f"不支持的 agent_name={agent_name!r}")

    require_textgrad()
    loaded = fixtures or load_mini_pipeline_cases()
    runner = MiniPipelineRunner(llm=executor_llm, locale=loaded.locale)

    templates: Dict[str, str] = dict(prompt_templates)
    best_template = system_prompt_template or templates.get(agent_name, "")
    if not best_template:
        raise ValueError(f"缺少 {agent_name} 的 system_prompt_template")
    templates[agent_name] = best_template

    train_cases = loaded.cases_for_split(train_split)
    constraints = agent_graph_constraints(agent_name)

    baseline_report = await evaluate_mini_pipeline_benchmark(
        runner,
        fixtures=loaded,
        split=dev_split,
        prompt_templates=templates,
    )
    best_dev_score = baseline_report.average_score
    steps: List[OptimizationStepRecord] = []

    for step in range(1, max_steps + 1):
        pipeline_failures = await collect_mini_pipeline_failures(
            runner,
            train_cases,
            prompt_templates=templates,
            failure_threshold=failure_threshold,
        )
        failure_cases = collect_agent_step_failures(
            pipeline_failures,
            agent_name=agent_name,
            step_failure_threshold=step_failure_threshold,
        )

        train_scores = []
        for case in train_cases:
            item = evaluate_mini_pipeline_case(
                runner,
                case,
                prompt_templates=templates,
            )
            train_scores.append(item.score.total)
        train_average = sum(train_scores) / len(train_scores) if train_scores else 0.0

        if not failure_cases:
            steps.append(
                OptimizationStepRecord(
                    step=step,
                    train_average=train_average,
                    dev_average=best_dev_score,
                    candidate_dev_average=best_dev_score,
                    accepted=False,
                    failure_count=0,
                    prompt_preview=best_template[:160],
                    optimizer=TEXTGRAD_AGENT_MINI_PIPELINE_OPTIMIZER_NAME,
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
        run_single_agent_graph_step(
            graph,
            failure_cases,
            constraints=constraints,
        )
        candidate_template = graph.read_optimized_prompt_template()

        candidate_templates = dict(templates)
        candidate_templates[agent_name] = candidate_template
        candidate_report = await evaluate_mini_pipeline_benchmark(
            runner,
            fixtures=loaded,
            split=dev_split,
            prompt_templates=candidate_templates,
        )
        candidate_dev = candidate_report.average_score
        accepted = should_accept_candidate(candidate_dev, best_dev_score, rollback=rollback)

        if accepted:
            best_template = candidate_template
            templates[agent_name] = candidate_template
            best_dev_score = candidate_dev

        steps.append(
            OptimizationStepRecord(
                step=step,
                train_average=train_average,
                dev_average=best_dev_score,
                candidate_dev_average=candidate_dev,
                accepted=accepted,
                failure_count=len(failure_cases),
                prompt_preview=candidate_template[:160],
                optimizer=TEXTGRAD_AGENT_MINI_PIPELINE_OPTIMIZER_NAME,
            )
        )

        if accepted and candidate_dev >= 0.999:
            break

    return OptimizationResult(
        best_prompt=best_template,
        baseline_dev_score=baseline_report.average_score,
        best_dev_score=best_dev_score,
        steps=steps,
        optimizer=TEXTGRAD_AGENT_MINI_PIPELINE_OPTIMIZER_NAME,
    )
