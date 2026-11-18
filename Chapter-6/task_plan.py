"""Build dependency-safe travel task plans for Chapter 6.

This module implements the planning stage between request classification and
domain-agent routing. It reuses the Chapter 2 fact-survey structure, the
Chapter 4 task-decomposition prompt, and the Chapter 4 dependency-analysis
implementation to produce identified tasks and a validated execution order.

Main capabilities:

- Extract given facts, facts to look up, derivations, and educated guesses.
- Decompose a travel request into tasks aligned with available capabilities.
- Analyze task inputs and outputs, validate dependencies, and order execution.

The module intentionally does not select domain agents. ``agent_routing.py``
uses the resulting task plan as its input, keeping planning and routing as
separate responsibilities.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List

from config import BOOK_ROOT
from fact_survey import FACTS_PROMPT, parse_fact_survey
from llm import invoke_chat


CHAPTER4_DIR = BOOK_ROOT / "Chapter-4"
if not CHAPTER4_DIR.is_dir():
    raise FileNotFoundError(f"Could not locate {CHAPTER4_DIR}")
if str(CHAPTER4_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER4_DIR))

from dependency_analysis import analyze_task_dependencies
from task_decomposition import decompose_task


async def survey_facts(llm: Any, user_query: str) -> Dict[str, Any]:
    """Extract the Chapter 2 fact-survey structure for a user request.

    Args:
        llm: Model used for the fact survey.
        user_query: Original user request.

    Returns:
        The structured fact-survey result, including the raw model response.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = FACTS_PROMPT.format(today=today, task=user_query.strip())
    response = await invoke_chat(llm, prompt)
    return parse_fact_survey(response).model_dump()


async def create_task_plan(
    llm: Any,
    user_query: str,
    pre_survey: Dict[str, Any],
    memories: List[Dict[str, Any]],
    agent_registry: Any,
) -> Dict[str, Any]:
    """Create the task and dependency plan before domain-agent routing.

    Args:
        llm: Model used for decomposition and dependency analysis.
        user_query: Original user request.
        pre_survey: Structured fact-survey result.
        memories: Relevant long-term memory records.
        agent_registry: Registry providing capability and skill descriptions.

    Returns:
        A task plan containing goals, dependency-safe task definitions, and
        execution order. Tasks do not yet contain a domain-agent assignment.
    """
    background_parts = [
        "【思维链预调查】",
        json.dumps(
            {key: value for key, value in pre_survey.items() if key != "raw_text"},
            ensure_ascii=False,
        ),
    ]
    if memories:
        background_parts.extend(
            ["【检索到的长期记忆】", json.dumps(memories, ensure_ascii=False)]
        )

    try:
        decomposition = await decompose_task(
            llm,
            user_query,
            background_info="\n".join(background_parts),
            capability_catalog=agent_registry.get_capability_catalog_text(),
        )
    except ValueError:
        decomposition = {
            "total_goal": user_query.strip(),
            "sub_steps": [user_query.strip()],
            "fallback": "single_task",
        }

    sub_steps = decomposition["sub_steps"]
    id_to_task = {
        f"T{index + 1}": task
        for index, task in enumerate(sub_steps)
    }
    if len(id_to_task) <= 1:
        execution_order = list(id_to_task)
        depends_map = {
            task_id: []
            for task_id in id_to_task
        }
    else:
        execution_order, depends_map = await analyze_task_dependencies(
            llm,
            id_to_task,
            agent_registry.get_agent_parameters_text(),
        )

    tasks = [
        {
            "task_id": task_id,
            "description": description,
            "depends_on": depends_map.get(task_id, []),
        }
        for task_id, description in id_to_task.items()
    ]
    return {
        "pre_survey": {
            key: value
            for key, value in pre_survey.items()
            if key != "raw_text"
        },
        "pre_survey_raw": pre_survey.get("raw_text", ""),
        "retrieved_memories": memories,
        "total_goal": decomposition["total_goal"],
        "plan_fallback": decomposition.get("fallback"),
        "tasks": tasks,
        "execution_order": execution_order,
    }
