"""Agent-B3：mini-pipeline 串联 slot 优化（仿 planner_pipeline 顺序 pipeline）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agents.mini_pipeline.fixtures import MiniPipelineFixtures
from agent_framework.optimization.agents.runtime import (
    TRAVEL_OPTIMIZABLE_AGENTS,
    default_agent_prompt_template,
)
from agent_framework.optimization.core.result import OptimizationResult
from agent_framework.optimization.optimizers.textgrad_agent.pipeline_optimize import (
    optimize_agent_prompt_mini_pipeline,
)

# 默认串联顺序：天气 → 酒店 → 航班（与 fixture case-1 一致）
DEFAULT_MINI_PIPELINE_SLOTS: tuple[str, ...] = (
    "WeatherAgent",
    "HotelAgent",
    "FlightAgent",
)


@dataclass
class MiniPipelineOptimizationOutput:
    """串联优化汇总：各 slot 的 OptimizationResult + 最终 prompt 快照。"""

    slots: List[str]
    results: Dict[str, OptimizationResult] = field(default_factory=dict)
    prompt_templates: Dict[str, str] = field(default_factory=dict)


def parse_mini_pipeline_slots(raw: str) -> List[str]:
    """解析 --slots：``default`` / ``all`` / 逗号分隔 Agent 名。"""
    normalized = (raw or "default").strip()
    lower = normalized.lower()
    if lower in ("default", "pipeline"):
        return list(DEFAULT_MINI_PIPELINE_SLOTS)
    if lower == "all":
        return list(TRAVEL_OPTIMIZABLE_AGENTS)

    selected: List[str] = []
    for token in normalized.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in TRAVEL_OPTIMIZABLE_AGENTS:
            valid = ", ".join(TRAVEL_OPTIMIZABLE_AGENTS)
            raise ValueError(f"不支持的 slot='{name}'，可选: {valid}, default, all")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("slots 不能为空")
    return selected


def _build_initial_prompt_templates(
    fixtures: MiniPipelineFixtures,
    *,
    slots: Optional[List[str]] = None,
) -> Dict[str, str]:
    """为 pipeline 中出现的 Agent 加载默认 / optimized 模板。"""
    agents = set(slots or TRAVEL_OPTIMIZABLE_AGENTS)
    for case in fixtures.cases:
        for step in case.steps:
            agents.add(step.agent_name)
    return {
        name: default_agent_prompt_template(name, locale=fixtures.locale)
        for name in agents
    }


async def run_mini_pipeline_optimization(
    *,
    slots: List[str],
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: MiniPipelineFixtures,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    step_failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
) -> MiniPipelineOptimizationOutput:
    """Agent-B3：按 slot 顺序串联优化各 Agent system_prompt。

    每个 slot 优化时，前面 slot 的最优 prompt 已冻结；rollback 用 mini-pipeline dev 分。
    生产 LangGraph 路径不改动。
    """
    if not slots:
        raise ValueError("slots 不能为空")

    current_templates = _build_initial_prompt_templates(fixtures, slots=slots)
    results: Dict[str, OptimizationResult] = {}

    for slot in slots:
        result = await optimize_agent_prompt_mini_pipeline(
            agent_name=slot,
            executor_llm=executor_llm,
            optimizer_llm=optimizer_llm,
            prompt_templates=current_templates,
            fixtures=fixtures,
            max_steps=max_steps,
            failure_threshold=failure_threshold,
            step_failure_threshold=step_failure_threshold,
            rollback=rollback,
            train_split=train_split,
            dev_split=dev_split,
            system_prompt_template=current_templates.get(slot),
        )
        current_templates[slot] = result.best_prompt
        results[slot] = result

    return MiniPipelineOptimizationOutput(
        slots=list(slots),
        results=results,
        prompt_templates=dict(current_templates),
    )
