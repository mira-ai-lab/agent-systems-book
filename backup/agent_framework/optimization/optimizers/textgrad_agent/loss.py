"""textgrad_agent 模式下的 loss 模板与期望标签。"""

from __future__ import annotations

import json
from typing import List

from agent_framework.optimization.agents.fixtures import SingleAgentCase
from agent_framework.optimization.agents.runtime import AGENT_REQUIRED_PLACEHOLDERS

AGENT_GRAPH_EVAL_INSTRUCTION = """You evaluate a travel sub-agent run against benchmark expectations.

The agent should call the expected tool with appropriate arguments and produce a helpful final response.
Compare the actual agent output with the expected specification below.
Be specific so the system_prompt can be improved for similar queries.
"""

# 向后兼容 B1 名称
FLIGHT_AGENT_GRAPH_EVAL_INSTRUCTION = AGENT_GRAPH_EVAL_INSTRUCTION

AGENT_GRAPH_ROLE_DESCRIPTIONS = [
    "actual sub-agent output",
    "single-agent benchmark expectation",
]

# 通用约束：所有 Agent 优化后只返回模板正文
AGENT_GRAPH_COMMON_CONSTRAINTS = [
    "Return only the revised system_prompt template, not analysis.",
]

# 各 Agent 必须保留的占位符约束（与 runtime.AGENT_REQUIRED_PLACEHOLDERS 对应）
AGENT_GRAPH_PLACEHOLDER_CONSTRAINTS: dict[str, List[str]] = {
    agent_name: [
        f"The optimized system_prompt must retain placeholder {token} for {agent_name}."
        for token in placeholders
    ]
    for agent_name, placeholders in AGENT_REQUIRED_PLACEHOLDERS.items()
}

# B1 兼容
FLIGHT_AGENT_GRAPH_CONSTRAINTS = (
    AGENT_GRAPH_PLACEHOLDER_CONSTRAINTS["FlightAgent"] + AGENT_GRAPH_COMMON_CONSTRAINTS
)


def agent_graph_constraints(agent_name: str) -> List[str]:
    """返回某 Agent 在 TextualGradientDescent 中使用的 constraints 列表。"""
    placeholders = AGENT_GRAPH_PLACEHOLDER_CONSTRAINTS.get(agent_name)
    if placeholders is None:
        raise ValueError(f"未知 agent_name={agent_name!r}")
    return placeholders + AGENT_GRAPH_COMMON_CONSTRAINTS


def build_single_agent_expectation_label(case: SingleAgentCase) -> str:
    """将单 Agent fixture 整理为 MultiFieldEvaluation 的 label 文本。"""
    lines = [
        f"case_id: {case.case_id}",
        f"agent_name: {case.agent_name}",
        f"query: {case.user_query}",
        f"expected_tool: {case.tool}",
        f"expected_tool_args: {json.dumps(case.tool_args, ensure_ascii=False)}",
        f"response_keywords: {case.response_keywords}",
    ]
    return "\n".join(lines)


def create_agent_graph_loss_fn(engine):
    """创建比较 Agent 输出与 benchmark 期望的 MultiFieldEvaluation。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    from textgrad.loss import MultiFieldEvaluation

    instruction = Variable(
        AGENT_GRAPH_EVAL_INSTRUCTION,
        requires_grad=False,
        role_description="single-agent graph evaluation instruction",
    )
    return MultiFieldEvaluation(
        instruction,
        AGENT_GRAPH_ROLE_DESCRIPTIONS,
        engine,
    )
