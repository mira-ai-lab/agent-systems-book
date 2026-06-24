"""TaskPlanner 同步桥接：在 textgrad StringBasedFunction 中复用现有 async 实现。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.planner_runtime import build_planner


class TaskPlannerSyncBridge:
    """将 ``TaskPlanner`` 三步包装为 sync 调用，避免复制 prompt 组装逻辑。"""

    def __init__(self, planner: TaskPlanner):
        self._planner = planner

    @classmethod
    def from_prompts(
        cls,
        *,
        executor_llm: ChatOpenAI,
        registry: Any,
        locale: str,
        decomposition_prompt: str,
        agent_routing: str,
    ) -> "TaskPlannerSyncBridge":
        planner = build_planner(
            executor_llm,
            registry,
            locale=locale,
            prompt_overrides={
                "decomposition_prompt": decomposition_prompt,
                "agent_routing": agent_routing,
            },
            use_optimized=False,
        )
        return cls(planner)

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    def run_decomposition(
        self,
        *,
        decomposition_prompt: str,
        user_query: str,
        pre_survey: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._planner.prompts = replace(
            self._planner.prompts,
            decomposition_prompt=decomposition_prompt,
        )
        return self._run_async(
            self._planner.run_decomposition(user_query, pre_survey, [])
        )

    def run_dependency_analysis(self, sub_steps: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        return self._run_async(self._planner.run_dependency_analysis(sub_steps))

    def route_to_agents(
        self,
        *,
        agent_routing: str,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        self._planner.prompts = replace(
            self._planner.prompts,
            agent_routing=agent_routing,
        )
        return self._run_async(
            self._planner.route_to_agents(sub_steps, execution_order, depends_map)
        )

    @staticmethod
    def format_pipeline_output(
        *,
        parsed: Dict[str, Any],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        routed_subtasks: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "totalGoal": parsed.get("totalGoal"),
            "subSteps": parsed.get("subSteps"),
            "execution_order": execution_order,
            "depends_map": depends_map,
            "routed_subtasks": routed_subtasks,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
