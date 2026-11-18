"""Route planned tasks to Chapter 6 domain agents.

This module is the independent agent-routing stage used after task planning. It
renders the registered agent capability catalog, asks the language model to
select one suitable agent per task, and extracts parameters accepted by that
agent's tool schema.

Main capabilities:

- Match each planned task to one registered domain agent.
- Fill tool parameters from the fact-survey result and original task context.
- Validate agent names and filter unsupported tool parameters.
- Reattach task descriptions and validated dependencies from local code.

The model supplies only routing decisions and tool parameters. The module keeps
task identity, dependency relationships, and execution order authoritative.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from llm import invoke_json


AGENT_ROUTING_PROMPT = """你是中心智能体的「子任务路由模块」。根据已拆解的子任务，为每个子任务选择最合适的领域智能体并提取调用参数。

## 可用领域智能体
{agent_team}

## 事实调查结果
{pre_survey_json}

## 子任务列表
{subtasks_json}

## 依赖分析结果
以下 depends_on 已通过本地校验，必须原样使用：
{depends_on_json}

## 输出要求
只输出 JSON 数组，其中每个元素对应一个子任务：
[
  {{
    "task_id": "T1",
    "agent": "WeatherAgent",
    "params": {{"city": "上海", "date": "2026-05-01"}}
  }}
]

规则：
1. agent 必须是可用领域智能体之一。
2. 每个子任务只分配一个 agent。
3. params 字段名必须匹配该 agent 的 inputSchema。
4. 必须为每个 task_id 输出一条记录。
5. 不要输出 description、depends_on 或其他解释文字，任务原文和依赖关系由本地代码保留。
"""


class AgentRouter:
    """Route subtasks to domain agents using the registered capabilities."""

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        """Initialize the router.

        Args:
            llm: Model used to select agents and extract tool parameters.
            agent_registry: Registry exposing agent capabilities and parameters.
        """
        self.llm = llm
        self.agent_registry = agent_registry

    def _valid_agent_names(self) -> set[str]:
        """Return the domain-agent names that routing may select."""
        agents = getattr(self.agent_registry, "agents", None)
        return set(agents.keys()) if isinstance(agents, dict) else set()

    @staticmethod
    def _merge_routed_item(
        task_id: str,
        item: Dict[str, Any],
        *,
        id_to_desc: Dict[str, str],
        depends_map: Dict[str, List[str]],
        valid_agents: set[str],
        allowed_fields: Dict[str, set[str]],
    ) -> Dict[str, Any]:
        """Validate one routed item and reattach authoritative dependencies.

        Args:
            task_id: Stable identifier assigned to the subtask.
            item: Raw routing object returned by the language model.
            id_to_desc: Mapping from task IDs to original descriptions.
            depends_map: Locally validated dependencies keyed by task ID.
            valid_agents: Domain-agent names accepted by the registry.
            allowed_fields: Tool parameter names accepted by each agent.

        Returns:
            A normalized routed subtask, including a routing error when the
            selected agent is unavailable.
        """
        agent = (item.get("agent") or "").strip()
        raw_params = item.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        if agent not in valid_agents:
            return {
                "task_id": task_id,
                "description": id_to_desc.get(task_id, ""),
                "agent": None,
                "params": params,
                "depends_on": depends_map.get(task_id, []),
                "routing_error": "invalid_agent",
            }
        accepted = allowed_fields.get(agent, set())
        filtered_params = {
            key: value
            for key, value in params.items()
            if key in accepted
        }
        return {
            "task_id": task_id,
            "description": id_to_desc.get(task_id, ""),
            "agent": agent,
            "params": filtered_params,
            "depends_on": depends_map.get(task_id, []),
            "ignored_params": sorted(set(params) - set(filtered_params)),
        }

    @staticmethod
    def _unrouted_subtask(
        task_id: str,
        *,
        id_to_desc: Dict[str, str],
        depends_map: Dict[str, List[str]],
        reason: str,
    ) -> Dict[str, Any]:
        """Represent a subtask that the model did not route.

        Args:
            task_id: Stable identifier assigned to the subtask.
            id_to_desc: Mapping from task IDs to original descriptions.
            depends_map: Locally validated dependencies keyed by task ID.
            reason: Machine-readable routing failure reason.

        Returns:
            A normalized subtask with no selected agent.
        """
        return {
            "task_id": task_id,
            "description": id_to_desc.get(task_id, ""),
            "agent": None,
            "params": {},
            "depends_on": depends_map.get(task_id, []),
            "routing_error": reason,
        }

    async def _invoke_routing_llm(
        self,
        subtasks_for_prompt: List[Dict[str, Any]],
        pre_survey: Dict[str, Any],
        depends_map: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Ask the model to route one batch of subtasks.

        Args:
            subtasks_for_prompt: Compact subtask records visible to the model.
            pre_survey: Structured facts extracted before decomposition.
            depends_map: Locally validated task dependencies.

        Returns:
            Non-empty routing records returned by the model.
        """
        pre_survey_data = {
            key: value
            for key, value in pre_survey.items()
            if key != "raw_text"
        }
        prompt = AGENT_ROUTING_PROMPT.format(
            agent_team=(
                self.agent_registry.get_all_agents_text()
                + "\n"
                + self.agent_registry.get_agent_parameters_text()
            ),
            pre_survey_json=json.dumps(
                pre_survey_data,
                ensure_ascii=False,
                indent=2,
            ),
            subtasks_json=json.dumps(subtasks_for_prompt, ensure_ascii=False, indent=2),
            depends_on_json=json.dumps(depends_map, ensure_ascii=False, indent=2),
        )
        routed = await invoke_json(self.llm, prompt)
        if not isinstance(routed, list):
            routed = routed.get("subtasks", []) if isinstance(routed, dict) else []
        return [
            item
            for item in routed
            if isinstance(item, dict) and item.get("task_id")
        ]

    async def route_to_agents(
        self,
        tasks: List[Dict[str, Any]],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        pre_survey: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Route every subtask to one domain agent.

        Args:
            tasks: Structured tasks from Chapter 4 planning.
            execution_order: Dependency-safe task IDs.
            depends_map: Locally validated dependencies keyed by task ID.
            pre_survey: Structured facts extracted before decomposition.

        Returns:
            Routed subtasks in execution order. Missing or invalid model output
            is represented with a ``routing_error`` field rather than dropped.
        """
        valid_agents = self._valid_agent_names()
        allowed_fields = {
            agent_name: self.agent_registry.get_agent_input_fields(agent_name)
            for agent_name in valid_agents
        }
        task_by_id = {
            task["task_id"]: task
            for task in tasks
        }
        id_to_desc = {
            task_id: task.get("description", "")
            for task_id, task in task_by_id.items()
        }

        def prompt_batch(task_ids: List[str]) -> List[Dict[str, Any]]:
            return [
                {
                    "task_id": task_id,
                    "description": id_to_desc.get(task_id, ""),
                    "depends_on": depends_map.get(task_id, []),
                }
                for task_id in task_ids
            ]

        by_id: Dict[str, Dict[str, Any]] = {}
        for item in await self._invoke_routing_llm(
            prompt_batch(execution_order),
            pre_survey,
            depends_map,
        ):
            by_id[item["task_id"]] = item

        missing = [task_id for task_id in execution_order if task_id not in by_id]
        if missing:
            for item in await self._invoke_routing_llm(
                prompt_batch(missing),
                pre_survey,
                depends_map,
            ):
                by_id[item["task_id"]] = item

        subtasks: List[Dict[str, Any]] = []
        for task_id in execution_order:
            if task_id not in by_id:
                subtasks.append(
                    self._unrouted_subtask(
                        task_id,
                        id_to_desc=id_to_desc,
                        depends_map=depends_map,
                        reason="llm_routing_missing",
                    )
                )
            else:
                subtasks.append(
                    self._merge_routed_item(
                        task_id,
                        by_id[task_id],
                        id_to_desc=id_to_desc,
                        depends_map=depends_map,
                        valid_agents=valid_agents,
                        allowed_fields=allowed_fields,
                    )
                )
        return subtasks
