"""任务规划：预调查 + 拆解 + 依赖 + 路由（DomainConfig 注入 context / guess）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.parsing import (
    parse_decomposition_response,
    parse_dependency_analysis,
    parse_json_from_llm,
    parse_pre_survey,
)
from agent_framework.infra.resilience.retry import async_retry, is_retryable_llm_error
from agent_framework.tracing.trace_provider import span_name, trace_span

from domains.travel.plan_context import build_agent_routing_format_kwargs


class TaskPlanner:
    def __init__(
        self,
        llm: ChatOpenAI,
        agent_registry: Any,
        prompts: DomainPrompts,
        domain_config: Optional[DomainConfig] = None,
    ):
        self.llm = llm
        self.agent_registry = agent_registry
        self.prompts = prompts
        self.domain_config = domain_config or DomainConfig()

    async def _ainvoke_llm(self, messages: list) -> Any:
        """LLM 调用（带可重试错误的指数退避）。"""

        async def _call():
            return await self.llm.ainvoke(messages)

        return await async_retry(
            _call,
            retry_on=(Exception,),
            should_retry=is_retryable_llm_error,
        )

    def _context_prefix(self) -> str:
        return self.domain_config.build_context_block()

    def _resolve_routed_agent(
        self,
        raw_agent: Optional[str],
        task_id: str,
        id_to_desc: Dict[str, str],
    ) -> Tuple[Optional[str], str]:
        resolved = self.agent_registry.resolve_agent(raw_agent)
        if resolved:
            return resolved, "llm"

        fallback = self.domain_config.routing_fallback
        if fallback:
            fb = self.agent_registry.resolve_agent(fallback)
            if fb:
                return fb, "routing_fallback"

        if self.domain_config.enable_guess_agent:
            guessed = self.domain_config.guess_agent(
                id_to_desc.get(task_id, ""),
                self.agent_registry,
            )
            if guessed:
                return guessed, "guess_agent"

        return None, "routing_failed"

    @trace_span(name=span_name("planner.pre_survey"), attrs_args=["user_query"])
    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        prompt = self.prompts.facts_prompt.format(task=user_query.strip())
        response = await self._ainvoke_llm([HumanMessage(content=prompt)])
        return parse_pre_survey(response.content or "")

    @trace_span(
        name=span_name("planner.decomposition"),
        attrs_args=["user_query"],
        record_result=False,
    )
    async def run_decomposition(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        background_parts = []
        ctx_block = self._context_prefix()
        if ctx_block:
            background_parts.append(ctx_block)
        background_parts.append("【思维链预调查】")
        background_parts.append(
            json.dumps({k: v for k, v in pre_survey.items() if k != "raw_text"}, ensure_ascii=False)
        )
        if memories:
            background_parts.append("【检索到的长期记忆】")
            background_parts.append(json.dumps(memories, ensure_ascii=False))

        prompt = self.prompts.decomposition_prompt.format(
            background_info="\n".join(background_parts),
            agent_team=self.agent_registry.get_all_agents_text(),
            user_input=user_query.strip(),
        )
        response = await self._ainvoke_llm([HumanMessage(content=prompt)])
        return parse_decomposition_response(response.content or "", lang="zh")

    @trace_span(
        name=span_name("planner.dependency"),
        attrs_args=["sub_steps"],
        record_result=False,
    )
    async def run_dependency_analysis(self, sub_steps: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        id_to_task = {f"T{i + 1}": task for i, task in enumerate(sub_steps)}
        user_prompt = self.prompts.dependency_user.format(
            subtasks=id_to_task,
            agents=self.agent_registry.get_agent_parameters_text(),
        )
        response = await self._ainvoke_llm([
            SystemMessage(content=self.prompts.dependency_system),
            HumanMessage(content=user_prompt),
        ])
        parsed = parse_json_from_llm(response.content or "{}")
        task_ids = list(id_to_task.keys())
        execution_order, depends_map = parse_dependency_analysis(parsed, task_ids)
        return execution_order, depends_map

    @trace_span(
        name=span_name("planner.routing"),
        attrs_args=["sub_steps"],
        record_result=False,
    )
    async def route_to_agents(
        self,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        id_to_desc = {f"T{i + 1}": desc for i, desc in enumerate(sub_steps)}
        subtasks_for_prompt = [
            {"task_id": tid, "description": id_to_desc.get(tid, ""), "depends_on": depends_map.get(tid, [])}
            for tid in execution_order
        ]
        ctx = self._context_prefix()
        agent_team = self.agent_registry.get_all_agents_text()
        if ctx:
            agent_team = ctx + "\n" + agent_team
        prompt = self.prompts.agent_routing.format(
            **build_agent_routing_format_kwargs(
                agent_team=agent_team + "\n" + self.agent_registry.get_agent_parameters_text(),
                subtasks_json=json.dumps(subtasks_for_prompt, ensure_ascii=False, indent=2),
            ),
        )
        response = await self._ainvoke_llm([HumanMessage(content=prompt)])
        routed = parse_json_from_llm(response.content or "[]")
        if not isinstance(routed, list):
            routed = routed.get("subtasks", [])
        by_id = {t["task_id"]: t for t in routed if isinstance(t, dict) and t.get("task_id")}
        subtasks = []
        for tid in execution_order:
            item = by_id.get(tid)
            raw_agent = item.get("agent") if item else None
            agent_name, routing_status = self._resolve_routed_agent(raw_agent, tid, id_to_desc)
            subtasks.append({
                "task_id": tid,
                "description": (item.get("description") if item else None) or id_to_desc.get(tid, ""),
                "agent": agent_name,
                "routing_status": routing_status,
                "params": (item.get("params") if item else None) or {},
                "depends_on": (item.get("depends_on") if item else None) or depends_map.get(tid, []),
            })
        return subtasks

    @trace_span(
        name=span_name("planner.build_plan"),
        attrs_args=["user_query"],
        record_result=False,
    )
    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        decomposition = await self.run_decomposition(user_query, pre_survey, memories)
        execution_order, depends_map = await self.run_dependency_analysis(decomposition["subSteps"])
        subtasks = await self.route_to_agents(decomposition["subSteps"], execution_order, depends_map)
        execution_order = [t["task_id"] for t in subtasks]
        return {
            "pre_survey": {k: v for k, v in pre_survey.items() if k != "raw_text"},
            "pre_survey_raw": pre_survey.get("raw_text", ""),
            "retrieved_memories": memories,
            "total_goal": decomposition["totalGoal"],
            "subtasks": subtasks,
            "execution_order": execution_order,
        }
