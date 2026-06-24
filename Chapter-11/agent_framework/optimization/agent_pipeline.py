"""5 个子 Agent 并列 / 顺序 system_prompt 优化 pipeline（Agent-B2）。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agents.fixtures import SingleAgentCaseFixtures
from agent_framework.optimization.agents.runtime import TRAVEL_OPTIMIZABLE_AGENTS
from agent_framework.optimization.core.result import OptimizationResult
from agent_framework.optimization.optimizers.textgrad_agent.optimize import optimize_agent_prompt_graph


@dataclass
class AgentOptimizationOutput:
    """多 Agent 优化汇总结果。"""

    agents: List[str]
    parallel: bool
    results: Dict[str, OptimizationResult] = field(default_factory=dict)


def parse_agent_slots(raw: str) -> List[str]:
    """解析 --agent 参数：``all`` 或逗号分隔的 Agent 名列表。"""
    normalized = (raw or "all").strip()
    if normalized.lower() == "all":
        return list(TRAVEL_OPTIMIZABLE_AGENTS)

    selected: List[str] = []
    for token in normalized.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in TRAVEL_OPTIMIZABLE_AGENTS:
            valid = ", ".join(TRAVEL_OPTIMIZABLE_AGENTS)
            raise ValueError(f"不支持的 agent='{name}'，可选: {valid}, all")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("agents 不能为空")
    return selected


async def _optimize_one_agent(
    agent_name: str,
    *,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: SingleAgentCaseFixtures,
    max_steps: int,
    failure_threshold: float,
    rollback: bool,
    train_split: str,
    dev_split: str,
) -> tuple[str, OptimizationResult]:
    """单 Agent 优化任务（供 gather 或顺序循环调用）。"""
    result = await optimize_agent_prompt_graph(
        agent_name=agent_name,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=max_steps,
        failure_threshold=failure_threshold,
        rollback=rollback,
        train_split=train_split,
        dev_split=dev_split,
    )
    return agent_name, result


async def run_agent_optimization(
    *,
    agents: List[str],
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    fixtures: SingleAgentCaseFixtures,
    max_steps: int = 10,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    parallel: bool = True,
) -> AgentOptimizationOutput:
    """Agent-B2：对多个子 Agent 做 system_prompt 优化。

    Args:
        parallel: True 时用 asyncio.gather 并列优化（默认）；False 时按 agents 顺序逐个优化。
    """
    if not agents:
        raise ValueError("agents 不能为空")

    common_kwargs = dict(
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=max_steps,
        failure_threshold=failure_threshold,
        rollback=rollback,
        train_split=train_split,
        dev_split=dev_split,
    )

    results: Dict[str, OptimizationResult] = {}

    if parallel and len(agents) > 1:
        # 并列：每个 Agent 独立跑完整优化循环，最后统一写盘避免并发写 JSON
        pairs = await asyncio.gather(
            *[_optimize_one_agent(name, **common_kwargs) for name in agents]
        )
        results = dict(pairs)
    else:
        # 顺序：降低 LLM 并发压力，行为与 planner_pipeline 的 slot 串联类似
        for name in agents:
            _, result = await _optimize_one_agent(name, **common_kwargs)
            results[name] = result

    return AgentOptimizationOutput(agents=list(agents), parallel=parallel, results=results)
