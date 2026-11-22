"""Reusable dependency analysis for later book chapters.

This module asks a model for subtask dependencies, validates the dependency
graph, and returns a dependency-safe execution order. It also hosts the
shared validators reused by conflict replanning and travel plan generation.
The Chapter 4 notebook does not import it.

:author: zhanghong, liuhui, wulin
:date: 2026-11-16
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from llm import invoke_json


DEPENDENCY_ANALYSIS_SYSTEM_PROMPT = """你是一个任务规划系统中的[任务依赖分析器], 负责根据子任务与候选能力的技能输入输出定义, 识别子任务之间的执行依赖后调整子任务的执行顺序

# 输入
1. subtasks: 已拆分的子任务列表, 每个子任务包含:
   - task_id（如 T1、T2 等）
   - task_description（子任务的自然语言描述）
2. capabilities: 候选能力列表, 每个能力提供若干技能, 每个技能包含:
   - skill_name
   - input_parameters
   - output_parameters

# 目标
你的目标是分析子任务之间的必要依赖关系后调整子任务的执行顺序
第一步：建立全局输入输出视图  
- 遍历所有子任务  
- 对每个子任务，判断最可能由哪项能力的哪个技能来完成  
- 读取并理解该技能的 输入参数 和 输出参数  
- 记录每个子任务的 [输入] 和 [输出]  

第二步：识别必要的任务依赖  
- 在获得所有子任务的输入输出信息后，从全局视角分析依赖关系 
- 如果子任务 A 输入参数中要求的信息，能够从子任务B的输出参数中获取，则A依赖B， B要先于A执行 
- 不要引入不确定或非必要的依赖关系

第三步：调整执行顺序  
- 仅当存在明确且必要的依赖关系时，才调整子任务顺序, 对不存在依赖关系的子任务，保持其原有的相对顺序  
- 调整后的顺序必须满足所有必要的依赖约束  
- 不要进行多余的重排

# 输出
- 输出必须是合法的 JSON。
- 仅输出调整后的子任务执行顺序, 使用如下格式:
{
  "1": "task_id",
  "2": "task_id",
  ...
}
其中order是执行顺序，从1开始递增; task_id是输入部分的子任务id, 必须与输入中的子任务对应

# 编排附加输出
在保持以上分析步骤和执行顺序要求的前提下，同时输出可执行的依赖映射：
上一节“仅输出执行顺序”的格式由下面的编排扩展格式覆盖。
{
  "depends_on": {"T1": [], "T2": ["T1"]},
  "execution_order": {"1": "T1", "2": "T2"}
}
- depends_on 必须包含全部 task_id。
- execution_order 使用 Notebook 中的编号映射格式。

请严格按照上述 JSON 格式返回，除此之外不要有其他输出。
"""

DEPENDENCY_ANALYSIS_USER_PROMPT = """subtasks:
{subtasks}

capabilities:
{capabilities}

请严格按照输出格式返回, 除此之外不要有其他输出"""


def build_dependency_analysis_prompt(
    subtasks: Dict[str, str],
    capability_catalog: str,
) -> str:
    """Build the dependency analysis user prompt.

    Args:
        subtasks: Mapping from task ID to natural-language description.
        capability_catalog: Available capabilities and skill schemas.

    Returns:
        A fully formatted dependency analysis prompt.
    """
    return DEPENDENCY_ANALYSIS_USER_PROMPT.format(
        subtasks=json.dumps(subtasks, ensure_ascii=False, indent=2),
        capabilities=capability_catalog.strip(),
    )


def build_topological_order(
    task_ids: List[str],
    dependencies: Dict[str, List[str]],
    preferred_order: List[str] | None = None,
) -> List[str]:
    """Build a dependency-safe order while preserving preferred tie ordering."""
    preferred = [
        task_id
        for task_id in (preferred_order or [])
        if task_id in task_ids
    ]
    tie_order = list(dict.fromkeys(preferred + task_ids))
    remaining = set(task_ids)
    completed: set[str] = set()
    ordered: List[str] = []

    while remaining:
        ready = [
            task_id
            for task_id in tie_order
            if task_id in remaining
            and all(dep in completed for dep in dependencies.get(task_id, []))
        ]
        if not ready:
            raise ValueError("Dependency graph contains a cycle or unresolved task")
        for task_id in ready:
            ordered.append(task_id)
            completed.add(task_id)
            remaining.remove(task_id)
    return ordered


def require_text(value: Any, field_name: str) -> str:
    """Return one stripped non-empty string or raise a descriptive error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def normalize_string_list(value: Any, field_name: str) -> List[str]:
    """Validate a list containing unique non-empty strings."""
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    normalized: List[str] = []
    for item in value:
        text = require_text(item, f"{field_name} item")
        if text in normalized:
            raise ValueError(f"{field_name} contains duplicate value: {text}")
        normalized.append(text)
    return normalized


def normalize_dependencies(
    value: Any,
    task_ids: List[str],
) -> Dict[str, List[str]]:
    """Validate a complete dependency mapping against the current tasks."""
    if not isinstance(value, dict):
        raise ValueError("depends_on must be an object")

    expected = set(task_ids)
    actual = {
        require_text(task_id, "depends_on task_id")
        for task_id in value
    }
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            "depends_on must exactly cover all tasks; "
            f"missing={missing}, unknown={unknown}"
        )

    dependencies: Dict[str, List[str]] = {}
    for task_id in task_ids:
        values = normalize_string_list(
            value[task_id],
            f"depends_on.{task_id}",
        )
        if task_id in values:
            raise ValueError(f"Task {task_id} cannot depend on itself")
        unknown_dependencies = sorted(set(values) - expected)
        if unknown_dependencies:
            raise ValueError(
                f"Task {task_id} references unknown dependencies: "
                f"{unknown_dependencies}"
            )
        dependencies[task_id] = values
    return dependencies


def read_preferred_order(value: Any, task_ids: List[str]) -> List[str]:
    """Read a model order as a tie-breaking hint, never as ground truth."""
    if value is None:
        return []
    if isinstance(value, list):
        raw_order = value
    elif isinstance(value, dict):
        def order_key(item: Any) -> tuple[int, Any]:
            text = str(item)
            return (0, int(text)) if text.isdigit() else (1, text)

        raw_order = [value[key] for key in sorted(value, key=order_key)]
    else:
        raise ValueError("execution_order must be a list or object")

    order = [
        require_text(task_id, "execution_order task_id")
        for task_id in raw_order
    ]
    if len(order) != len(set(order)):
        raise ValueError("execution_order contains duplicate task IDs")
    unknown = sorted(set(order) - set(task_ids))
    if unknown:
        raise ValueError(f"execution_order references unknown tasks: {unknown}")
    return order


def normalize_dependency_analysis(
    payload: Dict[str, Any],
    task_ids: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Validate model output and return an execution order and dependency map."""
    if not isinstance(payload, dict):
        raise ValueError("Dependency analysis response must be a JSON object")

    raw_dependencies = payload.get("depends_on")
    dependencies = normalize_dependencies(raw_dependencies, task_ids)

    raw_order = payload.get("execution_order")
    preferred_order = read_preferred_order(raw_order, task_ids)
    missing_order_ids = sorted(set(task_ids) - set(preferred_order))
    unknown_order_ids = sorted(set(preferred_order) - set(task_ids))
    if missing_order_ids or unknown_order_ids:
        raise ValueError(
            "execution_order must exactly cover all tasks; "
            f"missing={missing_order_ids}, unknown={unknown_order_ids}"
        )
    execution_order = build_topological_order(
        task_ids,
        dependencies,
        preferred_order,
    )
    return execution_order, dependencies


async def analyze_task_dependencies(
    llm: Any,
    subtasks: Dict[str, str],
    capability_catalog: str,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Analyze dependencies and return a validated execution plan.

    Args:
        llm: LangChain-compatible chat model.
        subtasks: Mapping from task ID to task description.
        capability_catalog: Available capability and skill definitions.

    Returns:
        A tuple containing the execution order and per-task dependencies.
    """
    if not subtasks:
        raise ValueError("subtasks must not be empty")

    prompt = build_dependency_analysis_prompt(subtasks, capability_catalog)
    payload = await invoke_json(
        llm,
        prompt,
        system_prompt=DEPENDENCY_ANALYSIS_SYSTEM_PROMPT,
    )
    return normalize_dependency_analysis(payload, list(subtasks))
