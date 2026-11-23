"""Prompt templates to be optimized in Chapter-10.

These are deliberately simple decomposition and routing templates that serve
as the starting point for TextGrad-based prompt self-evolution. Placeholders
are kept identical to those used by the rendering helpers in ``runtime`` and
``planner_graph``.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

import re
from typing import Tuple

DECOMPOSITION_PLACEHOLDERS = ("{background_info}", "{capability_catalog}", "{user_input}")
ROUTING_PLACEHOLDERS = ("{agent_team}", "{pre_survey_json}", "{subtasks_json}", "{depends_on_json}")

TASK_DECOMPOSITION_PROMPT = """你是任务拆解助手，请根据用户输入拆解任务。

背景信息：
{background_info}

可用能力：
{capability_catalog}

用户输入：
{user_input}

请按以下格式输出：
# 目标
一句话概括目标
# 任务拆解
- 子任务1
- 子任务2
"""

AGENT_ROUTING_PROMPT = """为每个子任务选择一个智能体，并输出 JSON 数组。

智能体列表：
{agent_team}

事实：
{pre_survey_json}

子任务：
{subtasks_json}

依赖：
{depends_on_json}

只输出 JSON 数组，格式如下：
[{{"task_id": "T1", "agent": "智能体名", "params": {{}}}}]
"""


def extract_prompt_template(raw: str, placeholders: Tuple[str, ...]) -> str:
    """Strip fences, verify placeholders, and reject unexpected format braces."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:markdown|text|prompt)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    missing = [placeholder for placeholder in placeholders if placeholder not in text]
    if missing:
        raise ValueError(f"optimized prompt template is missing placeholders: {missing}")
    probe = {placeholder.strip("{}"): "x" for placeholder in placeholders}
    try:
        text.format(**probe)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"optimized prompt template contains unexpected braces: {exc}") from exc
    return text
