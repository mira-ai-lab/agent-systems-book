"""Chapter-8: 任务规划模块 — 整合 Chapter-2 预调查 + Chapter-4 拆解与依赖分析"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from travel_multi_agent.domain.parsing import (
    guess_agent,
    order_from_dependency_json,
    parse_decomposition_response,
    parse_json_from_llm,
    parse_pre_survey,
)
from travel_multi_agent.domain.prompts import (
    AGENT_ROUTING_PROMPT,
    DEPENDENCY_SYSTEM_PROMPT_ZH,
    DEPENDENCY_USER_PROMPT_ZH,
    FACTS_PROMPT,
    PROMPT_TP_ZH,
)


class TaskPlanner:
    """整合 Ch2 + Ch4 + 子智能体路由的任务规划器。

    典型调用链：预调查 → 任务拆解 → 依赖排序 → Agent 路由 → build_execution_plan 汇总。
    """

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        """注入 LLM 与 Agent 注册表；注册表用于向 prompt 提供可用 Agent 及其参数说明。"""
        self.llm = llm
        self.agent_registry = agent_registry

    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        """Chapter-2 预调查：让 LLM 梳理已知事实、待查事实、待推导项与合理猜测。

        返回四段式结构化结果（given_facts / facts_to_lookup 等）及 raw_text 原文。
        """
        prompt = FACTS_PROMPT.format(task=user_query.strip())
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return parse_pre_survey(response.content or "")

    async def run_decomposition(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Chapter-4 任务拆解：结合预调查结论与长期记忆，将用户请求拆成可执行子任务。

        返回 {"totalGoal": 整体目标, "subSteps": 子任务描述列表}。
        """
        background_parts = [
            "【思维链预调查】",
            json.dumps({k: v for k, v in pre_survey.items() if k != "raw_text"}, ensure_ascii=False),
        ]
        if memories:
            background_parts.append("【检索到的长期记忆】")
            background_parts.append(json.dumps(memories, ensure_ascii=False))

        prompt = PROMPT_TP_ZH.format(
            background_info="\n".join(background_parts),
            agent_team=self.agent_registry.get_all_agents_text(),
            user_input=user_query.strip(),
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return parse_decomposition_response(response.content or "", lang="zh")

    async def run_dependency_analysis(self, sub_steps: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        """分析子任务执行顺序：为每个子任务分配 T1/T2/... 编号，由 LLM 输出依赖排序 JSON。

        返回 (execution_order, depends_map)。
        注意：当前 depends_map 均为空列表，依赖关系主要由 execution_order 体现；
        若 LLM 返回的排序与任务数不一致，会回退为默认顺序 T1, T2, ...
        """
        id_to_task = {f"T{i + 1}": task for i, task in enumerate(sub_steps)}
        user_prompt = DEPENDENCY_USER_PROMPT_ZH.format(
            subtasks=id_to_task,
            agents=self.agent_registry.get_agent_parameters_text(),
        )
        response = await self.llm.ainvoke([
            SystemMessage(content=DEPENDENCY_SYSTEM_PROMPT_ZH),
            HumanMessage(content=user_prompt),
        ])
        order_json = parse_json_from_llm(response.content or "{}")
        execution_order = order_from_dependency_json(order_json, len(sub_steps))

        depends_map: Dict[str, List[str]] = {tid: [] for tid in id_to_task}
        return execution_order, depends_map

    async def route_to_agents(
        self,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """将子任务路由到具体 Agent，并为每个任务填充调用参数。

        LLM 路由失败或遗漏某 task_id 时，用 guess_agent() 按描述关键词做启发式兜底。
        返回列表元素形如：{task_id, description, agent, params, depends_on}。
        """
        id_to_desc = {f"T{i + 1}": desc for i, desc in enumerate(sub_steps)}
        subtasks_for_prompt = [
            {"task_id": tid, "description": id_to_desc.get(tid, ""), "depends_on": depends_map.get(tid, [])}
            for tid in execution_order
        ]
        prompt = AGENT_ROUTING_PROMPT.format(
            agent_team=self.agent_registry.get_all_agents_text()
            + "\n"
            + self.agent_registry.get_agent_parameters_text(),
            subtasks_json=json.dumps(subtasks_for_prompt, ensure_ascii=False, indent=2),
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        routed = parse_json_from_llm(response.content or "[]")
        if not isinstance(routed, list):
            routed = routed.get("subtasks", [])
        by_id = {t["task_id"]: t for t in routed if isinstance(t, dict) and t.get("task_id")}
        subtasks = []
        for tid in execution_order:
            if tid in by_id:
                item = by_id[tid]
                subtasks.append({
                    "task_id": tid,
                    "description": item.get("description") or id_to_desc.get(tid, ""),
                    "agent": item.get("agent", "ItineraryAgent"),
                    "params": item.get("params") or {},
                    "depends_on": item.get("depends_on") or depends_map.get(tid, []),
                })
            else:
                subtasks.append({
                    "task_id": tid,
                    "description": id_to_desc.get(tid, ""),
                    "agent": guess_agent(id_to_desc.get(tid, "")),
                    "params": {},
                    "depends_on": depends_map.get(tid, []),
                })
        return subtasks

    @staticmethod
    def _guess_agent(description: str) -> str:
        """根据子任务描述关键词匹配 Agent 名称（供测试或外部调用）。"""
        return guess_agent(description)

    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """串联拆解 → 依赖分析 → Agent 路由，生成下游编排器可直接消费的全量执行计划。

        假定 pre_survey 已由 run_pre_survey 完成；本方法不再重复预调查。
        """
        decomposition = await self.run_decomposition(user_query, pre_survey, memories)
        execution_order, depends_map = await self.run_dependency_analysis(decomposition["subSteps"])
        subtasks = await self.route_to_agents(decomposition["subSteps"], execution_order, depends_map)
        return {
            "pre_survey": {k: v for k, v in pre_survey.items() if k != "raw_text"},
            "pre_survey_raw": pre_survey.get("raw_text", ""),
            "retrieved_memories": memories,
            "total_goal": decomposition["totalGoal"],
            "subtasks": subtasks,
            "execution_order": execution_order,
        }
