"""Router Engine：入口理解 → RoutingPlan。

L1 路由流水线（按配置开关逐段执行）::

    query + history
      → history_gate          判断历史是否相关
      → interaction_rewrite   多轮改写为独立任务句
      → extraction            抽取结构化 events
      → knowledge_routing     知识库 / 关键词候选 Agent
      → classification        LLM 分类候选 Agent
      → profile 决策          workflow vs adaptive
      → task_decomposition    （workflow）任务拆解 + 语义路由
      → instruction_build     （adaptive）单 Agent 指令构建
      → RoutingPlan
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_openai import ChatOpenAI

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.router.config import RouterConfig
from agent_framework.router.context.history import HistoryInput, normalize_history
from agent_framework.router.helpers import agent_skill_text, select_primary_candidate
from agent_framework.router.instruction_builder import run_instruction_build
from agent_framework.router.plan import AgentCandidate, RoutingPlan, RoutingStep
from agent_framework.router.pre_survey_bridge import pre_survey_from_routing_plan
from agent_framework.router.profile import (
    PROFILE_ADAPTIVE,
    PROFILE_WORKFLOW,
    resolve_profile_with_reason,
)
from agent_framework.router.stages.classification import run_classification
from agent_framework.router.stages.extraction import run_extraction
from agent_framework.router.stages.history_gate import run_history_gate
from agent_framework.router.stages.interaction_rewrite import run_interaction_rewrite
from agent_framework.router.stages.knowledge_routing import merge_agent_candidates, resolve_knowledge_candidates
from agent_framework.router.stages.semantic_routing import should_use_semantic_routing
from agent_framework.router.stages.task_decomposition import run_task_decomposition
from agent_framework.router.stream_events import (
    candidates_payload,
    router_plan_event,
    router_stage_event,
    steps_payload,
)
from agent_framework.tracing.trace_provider import span_name, trace_span


class RouterEngine:
    """企业路由引擎 L1：多轮改写 + 事件抽取 + classification + 指令构建。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        registry: SubAgentRegistry,
        *,
        config: Optional[RouterConfig] = None,
        domain: str = "",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.config = config or RouterConfig()
        self.domain = (domain or "").strip()

    @trace_span(name=span_name("router.route"), attrs_args=["query"], record_result=False)
    async def route(
        self,
        query: str,
        *,
        history: Optional[HistoryInput] = None,
        locale: Optional[str] = None,
        previous_step_info: str = "",
        force_profile: Optional[str] = None,
        tenant_id: str = "default",
    ) -> RoutingPlan:
        """非流式入口：消费 ``route_stream`` 并返回最终 ``RoutingPlan``。"""
        plan: Optional[RoutingPlan] = None
        async for event in self.route_stream(
            query,
            history=history,
            locale=locale,
            previous_step_info=previous_step_info,
            force_profile=force_profile,
            tenant_id=tenant_id,
        ):
            if event.get("type") == "router.plan":
                plan = event["_plan_obj"]
        if plan is None:
            raise RuntimeError("RouterEngine.route_stream 未产出 router.plan")
        return plan

    async def route_stream(
        self,
        query: str,
        *,
        history: Optional[HistoryInput] = None,
        locale: Optional[str] = None,
        previous_step_info: str = "",
        force_profile: Optional[str] = None,
        tenant_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """逐阶段 yield 结构化 Router 事件，最后 yield ``router.plan``。"""
        loc = (locale or self.config.locale or "zh").strip() or "zh"
        stages: list[str] = []
        working_query = query.strip()
        history_text = normalize_history(history)
        history_relevant: Optional[bool] = None
        events: List[str] = []
        knowledge_candidates: List[AgentCandidate] = []
        knowledge_meta: List[dict] = []

        # --- Stage 1: 历史门控 ---
        if history_text and self.config.enable_history_gate:
            history_relevant = await run_history_gate(
                self.llm,
                working_query,
                history_text,
                locale=loc,
            )
            stages.append("history_gate")
            yield router_stage_event(
                "history_gate",
                {"history_relevant": history_relevant},
            )
            if not history_relevant:
                history_text = ""

        # --- Stage 2: 多轮改写（依赖历史上下文补全当前句）---
        if history_text and self.config.enable_interaction_rewrite:
            working_query = await run_interaction_rewrite(
                self.llm,
                working_query,
                history_text,
                locale=loc,
                task_info=self.config.task_info,
            )
            stages.append("interaction_rewrite")
            yield router_stage_event(
                "interaction_rewrite",
                {"rewritten_query": working_query},
            )

        # --- Stage 3: 事件抽取（供知识路由 / 下游 Planner 使用）---
        if self.config.enable_extraction:
            events = await run_extraction(
                self.llm,
                working_query,
                locale=loc,
                history=history_text or None,
                note=self.config.extraction_note,
            )
            stages.append("extraction")
            yield router_stage_event("extraction", {"events": list(events)})

        # --- Stage 4: 知识库 / 规则候选 Agent（可与 LLM 分类结果合并）---
        if self.config.enable_knowledge_routing:
            knowledge_candidates, knowledge_meta = resolve_knowledge_candidates(
                self.registry,
                domain=self.domain,
                query=working_query,
                events=events,
                config=self.config,
                tenant_id=tenant_id,
            )
            if knowledge_candidates:
                stages.append("knowledge_routing")
                yield router_stage_event(
                    "knowledge_routing",
                    {
                        "candidates": candidates_payload(knowledge_candidates),
                        "matches": knowledge_meta,
                    },
                )

        # --- Stage 5: LLM 分类候选 Agent ---
        candidates: List[AgentCandidate] = []
        if self.config.enable_classification:
            candidates = await run_classification(
                self.llm,
                self.registry,
                working_query,
                locale=loc,
                note=self.config.classification_note,
            )
            stages.append("classification")
            yield router_stage_event(
                "classification",
                {"candidates": candidates_payload(candidates)},
            )

        if knowledge_candidates:
            candidates = merge_agent_candidates(knowledge_candidates, candidates)

        # --- Profile：≥2 Agent → workflow（Fixed Graph），否则 adaptive（Supervisor）---
        profile, profile_reason = resolve_profile_with_reason(
            candidates,
            force_profile=force_profile,
        )
        profile = profile.strip()

        primary = select_primary_candidate(candidates)
        primary_agent: Optional[str] = None
        agent_instruction: Optional[str] = None
        steps: List[RoutingStep] = []
        decomposition_goal = ""

        # --- Stage 6: workflow 路径 — 任务拆解（travel 可接语义 agent_routing）---
        if profile == PROFILE_WORKFLOW and self.config.enable_task_decomposition:
            router_pre_survey = None
            if should_use_semantic_routing(self.domain, self.config):
                # 将当前路由中间态转为 Planner 预调查输入，供拆解 / 路由 LLM 使用
                router_pre_survey = pre_survey_from_routing_plan(
                    RoutingPlan(
                        rewritten_query=working_query,
                        candidates=candidates,
                        events=events,
                        profile=profile,
                        primary_agent=primary.name if primary else None,
                        metadata={
                            "stages": list(stages),
                            "knowledge_matches": knowledge_meta,
                        },
                    )
                )
            decomposition_goal, steps = await run_task_decomposition(
                self.llm,
                self.registry,
                working_query,
                candidates,
                locale=loc,
                domain=self.domain,
                config=self.config,
                router_pre_survey=router_pre_survey,
            )
            stages.append("task_decomposition")
            if should_use_semantic_routing(self.domain, self.config):
                stages.append("semantic_routing")
            yield router_stage_event(
                "task_decomposition",
                {
                    "goal": decomposition_goal,
                    "steps": steps_payload(steps),
                },
            )

        # --- Stage 7: adaptive 路径 — 为首选 Agent 生成单步指令 ---
        if primary and self.config.enable_instruction_build and profile == PROFILE_ADAPTIVE:
            primary_agent = primary.name
            skill = agent_skill_text(self.registry.agents, primary.name)
            agent_instruction = await run_instruction_build(
                self.llm,
                init_task=working_query,
                target_agent=primary.name,
                agent_skill=skill,
                previous_step_info=previous_step_info,
                locale=loc,
            )
            stages.append("instruction_build")
            yield router_stage_event(
                "instruction_build",
                {
                    "primary_agent": primary_agent,
                    "agent_instruction": agent_instruction,
                },
            )

        plan = RoutingPlan(
            rewritten_query=working_query,
            candidates=candidates,
            events=events,
            steps=steps,
            profile=profile,
            locale=loc,
            history_relevant=history_relevant,
            primary_agent=primary_agent,
            agent_instruction=agent_instruction,
            metadata={
                "router_version": "0.20",
                "forced_profile": force_profile,
                "profile_reason": profile_reason,
                "kb_tenant_id": tenant_id,
                "stages": stages,
                "decomposition_goal": decomposition_goal,
                "knowledge_matches": knowledge_meta
                or [
                    {"name": c.name, "score": c.score, "source": "keyword"}
                    for c in knowledge_candidates
                ],
            },
        )
        yield router_plan_event(plan)
