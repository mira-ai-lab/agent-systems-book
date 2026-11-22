"""Reusable task decomposition logic for later book chapters.

This module contains the agent-based task decomposition prompt and parser used
by Chapter 6. The Chapter 4 notebook intentionally stays capability-based and
does not import this module.

:author: zhanghong, liuhui, wulin
:date: 2026-11-16
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from llm import invoke_chat


TASK_DECOMPOSITION_PROMPT = """
##角色：你是一个高度熟练的任务拆解专家，专注于将用户输入拆解为粒度适当的子任务。
##任务：根据用户的输入，分析目标以及能力目录中所有可用能力的描述，通过将用户输入分解为能力目录可以覆盖的信息明确的子任务，以帮助有效地实现用户的目标。
##背景信息（关于具体任务拆解所需要知道的相关信息及要求）：
    {background_info}
##可用能力目录（可用于完成子任务的候选能力）：
    {capability_catalog}
##步骤：
    1. **理解目标**：根据所有用户输入确定用户想要实现的核心目标和结果。
    2. **考虑背景信息**：提供的背景信息中包含拆解任务所需要的额外的领域知识，可用作参考。
    3. **定义子任务**：根据目标的复杂程度将任务拆解为子任务。
      3.1 **任务拆解原则**：
        - 简单：每个子任务都是一个能够匹配能力目录中单一能力的祈使句指令。
        - 信息完整：每个子任务指令中必须包含明确的槽位信息，所需槽位信息从用户输入及目标中抽取。
        - 优化复杂度：根据任务复杂度调整子任务数量，使用最少子任务完成整体目标。
      3.2 **子任务格式**：不同子任务间使用#分割。
    4. **确认子任务能否完成**：逐个判断每个子任务能否由能力目录中的某项能力完成；只有某个子任务确实无法覆盖时，才将该子任务标记为'NULL'，能覆盖的子任务必须保留，不得把全部子任务统一标记为'NULL'。
    5. **补充子任务信息**：针对每个子任务，提取该子任务的所有槽位，并根据目标补全所有槽位，合并成一句完整的祈使指令生成新的子任务并替换原有子任务，使子任务内容信息明确指令清晰。
##要求：
    1.需要根据完整的用户输入的语义（包含所有输入内容）进行分析；
    2.背景信息仅作为参考，只有当需要时才使用；
    3.子任务仅能依靠能力目录中声明的技能完成，避免擅自扩展为目录之外的能力并创造无法完成的子任务。
    4.仅输出简洁且信息完整的整体目标内容以及槽位信息完整的任务拆解结果，不要输出分析过程及原因;
    5.子任务中必须包含明确完整的槽位信息，避免直接将智能体能力作为子任务输出；
    6.输出的子任务应为信息明确的完整指令，确保用户输入及目标中包含的所有信息都出现在对应在子任务中，避免槽位信息丢失或单独输出槽位信息作为子任务；
    7.能力目录中的不同能力可以组合完成复杂目标。某一个子任务不可覆盖时，只处理该子任务，不能用'NULL'否定其他可完成子任务。
##输出格式：
    # 目标
        整体目标内容（不包含人称的祈使句式）
    # 任务拆解
     - 子任务1
     - 子任务2(如有)
     - ...
##用户输入：
    {user_input}
"""


def build_task_decomposition_prompt(
    user_query: str,
    capability_catalog: str,
    *,
    background_info: str = "",
) -> str:
    """Build the task decomposition user prompt.

    Args:
        user_query: User request to decompose.
        capability_catalog: Available capability descriptions.
        background_info: Optional pre-survey and memory context.

    Returns:
        A fully formatted task decomposition prompt.
    """
    return TASK_DECOMPOSITION_PROMPT.format(
        background_info=background_info.strip(),
        capability_catalog=capability_catalog.strip(),
        user_input=user_query.strip(),
    )


def parse_task_decomposition_response(response: str) -> Dict[str, Any]:
    """Parse a Chinese task decomposition response.

    Args:
        response: Raw model response containing total goal and subtasks.

    Returns:
        A dictionary with ``total_goal`` and ``sub_steps``.
    """
    goal_lines: List[str] = []
    sub_steps: List[str] = []
    reached_tasks = False

    for raw_line in (response or "").splitlines():
        line = raw_line.strip()
        if not reached_tasks:
            if line.startswith("# 任务拆解"):
                reached_tasks = True
                continue
            if line.startswith("# 目标"):
                continue
            if line:
                goal_lines.append(line)
            continue

        if line.startswith(("-", "*")):
            task = re.sub(r"^[-*]\s+", "", line).strip()
            task = task.rstrip("#").strip()
            if (
                task
                and task.upper() != "NULL"
                and not re.search(r"[（(]\s*NULL\s*[）)]$", task, re.IGNORECASE)
            ):
                sub_steps.append(task)

    return {
        "total_goal": " ".join(goal_lines),
        "sub_steps": sub_steps,
    }


async def decompose_task(
    llm: Any,
    user_query: str,
    *,
    capability_catalog: str,
    background_info: str = "",
) -> Dict[str, Any]:
    """Decompose one user request into executable subtasks.

    Args:
        llm: LangChain-compatible chat model.
        user_query: User request to decompose.
        capability_catalog: Available capabilities and skill descriptions.
        background_info: Optional pre-survey and memory context.

    Returns:
        A dictionary with ``total_goal`` and non-empty ``sub_steps``.

    Raises:
        ValueError: If the response contains no valid subtask.
    """
    prompt = build_task_decomposition_prompt(
        user_query,
        capability_catalog,
        background_info=background_info,
    )
    response = await invoke_chat(llm, prompt)
    result = parse_task_decomposition_response(response)
    if not result["sub_steps"]:
        raise ValueError("Task decomposition returned no valid subtasks")
    return result
