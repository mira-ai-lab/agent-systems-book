"""Conflict-driven incremental replanning for the Chapter 4 case.

This module extends the task decomposition and dependency planning flow while
preserving the existing module interfaces. It asks the model to analyze one
runtime conflict, then validates the returned task changes and rebuilds a
dependency-safe execution order in code.

:author: wulin
:date: 2026-11-21
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from dependency_analysis import build_topological_order, normalize_dependencies, normalize_string_list, read_preferred_order, require_text
from llm import invoke_json


CONFLICT_REPLANNING_SYSTEM_PROMPT = """你是一个任务规划系统中的[冲突分析与重规划器]，负责在运行时事件使原计划失效后，识别受影响的任务，并在不改变未受影响任务的前提下尝试生成修订计划。

# 输入
1. initial_plan：已经通过依赖排序的初始计划，包含总体目标、子任务、依赖关系和执行顺序。
2. conflict_event：执行阶段新出现的冲突或约束变化。
3. capabilities：当前系统能够使用的候选能力及其输入输出定义。

# 处理步骤
1. 分析冲突事件影响了哪些原始任务，以及违反了哪些目标或约束。
2. 优先进行局部重规划：保留所有未受影响任务的 task_id 和原始描述，只修改、删除受影响任务，或新增确有必要的任务。
3. 根据修订后的任务重新给出直接依赖和建议执行顺序，不得引用不存在的 task_id。
4. 如果在现有能力、总体目标和冲突条件下无法形成可执行计划，不得虚构解决方案，必须返回 infeasible，并指出需要用户决定的事项。

# 可解决时的输出
{
  "status": "replanned",
  "conflict_analysis": {
    "summary": "冲突摘要",
    "affected_task_ids": ["T2"],
    "violated_constraints": ["被违反的约束"]
  },
  "revised_subtasks": {
    "T1": "未受影响的原任务，必须原样保留",
    "T2": "修订后的任务",
    "R1": "必要时新增的任务"
  },
  "depends_on": {
    "T1": [],
    "T2": ["R1"],
    "R1": ["T1"]
  },
  "execution_order": {"1": "T1", "2": "R1", "3": "T2"},
  "resolution_summary": "重规划结果摘要"
}

- revised_subtasks 必须给出修订后的完整任务集合。
- affected_task_ids 只能引用初始计划中的任务。
- 未列入 affected_task_ids 的原任务必须保留相同 task_id 和完全相同的任务描述。
- depends_on 必须覆盖 revised_subtasks 中的全部任务。
- execution_order 是建议顺序，程序会再次进行拓扑排序，不得用顺序掩盖依赖错误。

# 无法解决时的输出
{
  "status": "infeasible",
  "conflict_analysis": {
    "summary": "冲突摘要",
    "affected_task_ids": ["T2"],
    "violated_constraints": ["相互冲突的约束"]
  },
  "reason": "在现有条件下无法解决的原因",
  "unresolved_constraints": ["仍未解决的目标或约束"],
  "requires_user_decision": true,
  "suggested_actions": ["请用户调整日期", "请用户放宽某项约束"]
}

- unresolved_constraints 只列出直接构成当前冲突、且尚未满足的目标或约束；不要加入与冲突无关、仍可满足或仅待后续核验的预算、步数等条件。

请严格输出以上两种合法 JSON 之一，除此之外不要有其他输出。
"""


CONFLICT_REPLANNING_USER_PROMPT = """initial_plan:
{initial_plan}

conflict_event:
{conflict_event}

capabilities:
{capabilities}

请分析冲突并进行局部重规划；若现有条件下无法解决，返回 infeasible。"""






def _normalize_subtasks(value: Any, field_name: str = "subtasks") -> Dict[str, str]:
    """Validate a task-id to task-description mapping."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a non-empty object")

    normalized: Dict[str, str] = {}
    for raw_task_id, raw_description in value.items():
        task_id = require_text(raw_task_id, f"{field_name} task_id")
        description = require_text(
            raw_description,
            f"{field_name}.{task_id}",
        )
        normalized[task_id] = description
    return normalized






def _normalize_conflict_analysis(
    value: Any,
    original_task_ids: List[str],
) -> Dict[str, Any]:
    """Validate the common conflict-analysis block returned by the model."""
    if not isinstance(value, dict):
        raise ValueError("conflict_analysis must be an object")

    summary = require_text(value.get("summary"), "conflict_analysis.summary")
    affected = normalize_string_list(
        value.get("affected_task_ids"),
        "conflict_analysis.affected_task_ids",
    )
    if not affected:
        raise ValueError("conflict_analysis.affected_task_ids must not be empty")
    unknown = sorted(set(affected) - set(original_task_ids))
    if unknown:
        raise ValueError(f"affected_task_ids references unknown tasks: {unknown}")
    violated = normalize_string_list(
        value.get("violated_constraints", []),
        "conflict_analysis.violated_constraints",
    )
    return {
        "summary": summary,
        "affected_task_ids": affected,
        "violated_constraints": violated,
    }


def build_initial_plan(
    user_query: str,
    total_goal: str,
    subtasks: Dict[str, str],
    dependencies: Dict[str, List[str]] | None = None,
    execution_order: Any = None,
) -> Dict[str, Any]:
    """Build a versioned initial plan from validated task-planning outputs.

    ``execution_order`` may be the numbered JSON object returned by the model
    or a task-id list. It is used only as a tie-breaking preference; the final
    order is always rebuilt from ``dependencies`` in code.
    """
    query = require_text(user_query, "user_query")
    goal = require_text(total_goal, "total_goal")
    normalized_subtasks = _normalize_subtasks(subtasks)
    task_ids = list(normalized_subtasks)
    raw_dependencies = dependencies or {
        task_id: [] for task_id in task_ids
    }
    normalized_dependencies = normalize_dependencies(
        raw_dependencies,
        task_ids,
    )
    preferred_order = read_preferred_order(execution_order, task_ids)
    safe_order = build_topological_order(
        task_ids,
        normalized_dependencies,
        preferred_order,
    )
    return {
        "version": 1,
        "status": "ready",
        "user_query": query,
        "total_goal": goal,
        "subtasks": normalized_subtasks,
        "depends_on": normalized_dependencies,
        "execution_order": safe_order,
        "sorted_steps": [normalized_subtasks[task_id] for task_id in safe_order],
    }


def build_conflict_replanning_prompt(
    initial_plan: Dict[str, Any],
    conflict_event: str,
    capability_catalog: str,
) -> str:
    """Build the user prompt for one runtime conflict."""
    event = require_text(conflict_event, "conflict_event")
    capabilities = require_text(capability_catalog, "capability_catalog")
    return CONFLICT_REPLANNING_USER_PROMPT.format(
        initial_plan=json.dumps(initial_plan, ensure_ascii=False, indent=2),
        conflict_event=event,
        capabilities=capabilities,
    )




def normalize_replanning_response(
    payload: Dict[str, Any],
    initial_plan: Dict[str, Any],
    conflict_event: str,
) -> Dict[str, Any]:
    """Validate a model response and return a safe replanning result."""
    if not isinstance(payload, dict):
        raise ValueError("Replanning response must be a JSON object")
    if not isinstance(initial_plan, dict):
        raise ValueError("initial_plan must be an object")

    original_subtasks = _normalize_subtasks(initial_plan.get("subtasks"))
    original_task_ids = list(original_subtasks)
    event = require_text(conflict_event, "conflict_event")
    status = require_text(payload.get("status"), "status")
    if status not in {"replanned", "infeasible"}:
        raise ValueError("status must be replanned or infeasible")

    analysis = _normalize_conflict_analysis(
        payload.get("conflict_analysis"),
        original_task_ids,
    )
    affected = set(analysis["affected_task_ids"])

    if status == "infeasible":
        reason = require_text(payload.get("reason"), "reason")
        unresolved = normalize_string_list(
            payload.get("unresolved_constraints"),
            "unresolved_constraints",
        )
        if not unresolved:
            raise ValueError("unresolved_constraints must not be empty")
        if payload.get("requires_user_decision") is not True:
            raise ValueError("requires_user_decision must be true")
        suggested_actions = normalize_string_list(
            payload.get("suggested_actions"),
            "suggested_actions",
        )
        if not suggested_actions:
            raise ValueError("suggested_actions must not be empty")
        return {
            "status": "infeasible",
            "conflict_event": event,
            "conflict_analysis": analysis,
            "reason": reason,
            "unresolved_constraints": unresolved,
            "requires_user_decision": True,
            "suggested_actions": suggested_actions,
        }

    revised_subtasks = _normalize_subtasks(
        payload.get("revised_subtasks"),
        "revised_subtasks",
    )
    for task_id, description in original_subtasks.items():
        if task_id in affected:
            continue
        if task_id not in revised_subtasks:
            raise ValueError(f"Unaffected task {task_id} cannot be removed")
        if revised_subtasks[task_id] != description:
            raise ValueError(f"Unaffected task {task_id} cannot be changed")

    revised_task_ids = list(revised_subtasks)
    dependencies = normalize_dependencies(
        payload.get("depends_on"),
        revised_task_ids,
    )
    preferred_order = read_preferred_order(
        payload.get("execution_order"),
        revised_task_ids,
    )
    safe_order = build_topological_order(
        revised_task_ids,
        dependencies,
        preferred_order,
    )
    original_ids = set(original_subtasks)
    revised_ids = set(revised_subtasks)
    changes = [
        {"task_id": task_id, "change_type": "removed", "reason": "因冲突被移除"}
        for task_id in sorted(original_ids - revised_ids)
    ]
    changes.extend(
        {"task_id": task_id, "change_type": "added", "reason": "因冲突新增"}
        for task_id in sorted(revised_ids - original_ids)
    )
    changes.extend(
        {
            "task_id": task_id,
            "change_type": "updated",
            "reason": "因冲突修改",
        }
        for task_id in sorted(original_ids & revised_ids)
        if original_subtasks[task_id] != revised_subtasks[task_id]
    )
    summary = require_text(
        payload.get("resolution_summary"),
        "resolution_summary",
    )

    try:
        current_version = int(initial_plan.get("version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError("initial_plan.version must be an integer") from error

    revised_plan = {
        "version": current_version + 1,
        "status": "ready",
        "user_query": require_text(
            initial_plan.get("user_query"),
            "initial_plan.user_query",
        ),
        "total_goal": require_text(
            initial_plan.get("total_goal"),
            "initial_plan.total_goal",
        ),
        "subtasks": revised_subtasks,
        "depends_on": dependencies,
        "execution_order": safe_order,
        "sorted_steps": [revised_subtasks[task_id] for task_id in safe_order],
    }
    return {
        "status": "replanned",
        "conflict_event": event,
        "conflict_analysis": analysis,
        "revised_plan": revised_plan,
        "changes": changes,
        "resolution_summary": summary,
    }


async def replan_after_conflict(
    llm: Any,
    initial_plan: Dict[str, Any],
    conflict_event: str,
    capability_catalog: str,
) -> Dict[str, Any]:
    """Analyze one live conflict and return a validated replanning result."""
    prompt = build_conflict_replanning_prompt(
        initial_plan,
        conflict_event,
        capability_catalog,
    )
    payload = await invoke_json(
        llm,
        prompt,
        system_prompt=CONFLICT_REPLANNING_SYSTEM_PROMPT,
    )
    return normalize_replanning_response(
        payload,
        initial_plan,
        conflict_event,
    )
