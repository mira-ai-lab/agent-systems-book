"""RouterOrchestrator：profile=auto 时先路由再委托执行后端。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, Optional

from langchain_openai import ChatOpenAI

from agent_framework.config import load_project_dotenv
from agent_framework.domain.dynamic_registry import resolve_domain_registry_and_a2a
from agent_framework.domain.plugin import DomainPlugin
from agent_framework.infra.agent_runtime import configure_agent_llm
from agent_framework.i18n.agent_locale_context import agent_locale_context
from agent_framework.observability.request_context import request_metrics_context
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from agent_framework.orchestration.protocol import (
    MODE_FIXED_GRAPH,
    MODE_SUPERVISOR,
    OrchestrationBackend,
    TRANSPORT_LOCAL,
    AgentTransport,
)
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator
from agent_framework.orchestration.thread_stage_context import get_thread_stage_store
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine
from agent_framework.router.execution_plan_bridge import (
    ensure_execution_plan_from_routing_plan,
    enrich_execution_plan_pipeline_metadata,
)
from agent_framework.router.observability import enrich_routing_observability
from agent_framework.router.profile import PROFILE_AUTO, PROFILE_WORKFLOW, profile_to_mode
from agent_framework.stream.events import final_event, public_event
from agent_framework.tracing import get_logger, log_info, setup_observability, trace_span
from agent_framework.tracing.trace_provider import span_name

logger = get_logger(__name__)

MODE_ROUTER = "router"


class RouterOrchestrator:
    """企业路由入口：RouterEngine → workflow / adaptive 执行 Profile。"""

    mode = MODE_ROUTER

    def __init__(
        self,
        llm: ChatOpenAI,
        plugin: DomainPlugin,
        *,
        domain: str,
        user_id: str = "default",
        enable_memory: bool = True,
        enable_guess_agent: bool = True,
        long_term_backend: Optional[Any] = None,
        transport: AgentTransport = TRANSPORT_LOCAL,
        router_config: Optional[RouterConfig] = None,
        locale: str = "zh",
        entry_profile: str = PROFILE_AUTO,
    ) -> None:
        load_project_dotenv()
        setup_observability()
        self.domain = domain
        self.user_id = user_id
        self.llm = llm
        self.plugin = plugin
        self.enable_memory = enable_memory
        self.enable_guess_agent = enable_guess_agent
        self.long_term_backend = long_term_backend
        self.transport = transport
        self.locale = locale
        self.entry_profile = entry_profile
        self.registry, self._a2a_endpoints = resolve_domain_registry_and_a2a(domain, plugin)
        configure_agent_llm(llm)
        self._router = RouterEngine(llm, self.registry, config=router_config, domain=domain)
        self._workflow_backend: Optional[LangGraphOrchestrator] = None
        self._adaptive_backend: Optional[SupervisorOrchestrator] = None
        self._backend_lock = asyncio.Lock()

    def _router_unified_pipeline(self):
        cached = getattr(self, "_cached_router_pipeline", None)
        if cached is not None:
            return cached
        base = self.plugin.build_pipeline(enable_memory=self.enable_memory)
        self._cached_router_pipeline = replace(base, allow_task_planner_decomposition=False)
        return self._cached_router_pipeline

    async def _get_backend(self, exec_mode: str) -> OrchestrationBackend:
        async with self._backend_lock:
            if exec_mode == MODE_FIXED_GRAPH:
                if self._workflow_backend is None:
                    self._workflow_backend = LangGraphOrchestrator(
                        llm=self.llm,
                        domain=self.domain,
                        user_id=self.user_id,
                        registry=self.registry,
                        prompts=self.plugin.create_prompts(locale=self.locale).with_platform_defaults(self.locale),
                        domain_config=self.plugin.create_domain_config(
                            enable_guess_agent=self.enable_guess_agent
                        ),
                        pipeline=self._router_unified_pipeline(),
                        enable_guess_agent=self.enable_guess_agent,
                        long_term_backend=self.long_term_backend,
                        locale=self.locale,
                    )
                return self._workflow_backend

            if exec_mode == MODE_SUPERVISOR:
                if not self.plugin.supports_mode(MODE_SUPERVISOR):
                    raise ValueError(
                        f"领域 '{self.domain}' 不支持 supervisor，"
                        f"routing 结果无法使用 adaptive profile"
                    )
                if self._adaptive_backend is None:
                    self._adaptive_backend = SupervisorOrchestrator(
                        self.llm,
                        domain=self.domain,
                        user_id=self.user_id,
                        registry=self.registry,
                        prompts=self.plugin.create_prompts(locale=self.locale).with_platform_defaults(self.locale),
                        pipeline=self.plugin.build_pipeline(enable_memory=self.enable_memory),
                        enable_memory=self.enable_memory,
                        long_term_backend=self.long_term_backend,
                        transport=self.transport,
                        a2a_endpoints=self._a2a_endpoints,
                        locale=self.locale,
                    )
                return self._adaptive_backend

        raise ValueError(f"不支持的执行 mode='{exec_mode}'")

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
        timeout_sec: Optional[float] = None,
        *,
        conversation_history: Optional[str] = None,
    ) -> Dict[str, Any]:
        log_info(
            logger,
            "router.request.start",
            domain=self.domain,
            entry_profile=self.entry_profile,
            thread_id=thread_id,
            query_preview=user_query.strip()[:120],
        )
        force_profile = PROFILE_WORKFLOW if self.entry_profile == PROFILE_WORKFLOW else None
        with agent_locale_context(self.locale):
            plan = await self._router.route(
                user_query,
                history=conversation_history,
                locale=self.locale,
                previous_step_info=get_thread_stage_store().get_last_stage_summary(
                    self.domain,
                    thread_id,
                ),
                force_profile=force_profile,
                tenant_id=self.user_id,
            )
            if self.entry_profile == PROFILE_WORKFLOW:
                plan.profile = PROFILE_WORKFLOW

            exec_mode = (
                MODE_FIXED_GRAPH
                if self.entry_profile == PROFILE_WORKFLOW
                else profile_to_mode(plan.profile)
            )
            log_info(
                logger,
                "router.plan",
                entry_profile=self.entry_profile,
                profile=plan.profile,
                exec_mode=exec_mode,
                candidates=[(c.name, c.score) for c in plan.candidates],
                history_relevant=plan.history_relevant,
                primary_agent=plan.primary_agent,
            )

            with request_metrics_context(
                domain=self.domain,
                mode=exec_mode,
                transport=self.transport,
            ):
                backend = await self._get_backend(exec_mode)
                backend_kwargs: Dict[str, Any] = {}
                if exec_mode == MODE_FIXED_GRAPH:
                    pipeline = self._router_unified_pipeline()
                    backend_kwargs["prefilled_execution_plan"] = enrich_execution_plan_pipeline_metadata(
                        ensure_execution_plan_from_routing_plan(
                            plan,
                            user_query=plan.execution_query,
                        ),
                        pre_survey_mode=pipeline.pre_survey_mode,
                    )
                result = await backend.process_request(
                    plan.execution_query,
                    thread_id=thread_id,
                    timeout_sec=timeout_sec,
                    **backend_kwargs,
                )

        result["routing_plan"] = plan.to_dict()
        result["profile"] = self.entry_profile
        result["resolved_profile"] = plan.profile
        result["orchestration_mode"] = exec_mode
        enrich_routing_observability(result, domain=self.domain)
        return result

    async def iter_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
        *,
        conversation_history: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        log_info(
            logger,
            "router.request.stream.start",
            domain=self.domain,
            entry_profile=self.entry_profile,
            thread_id=thread_id,
            query_preview=user_query.strip()[:120],
        )
        force_profile = PROFILE_WORKFLOW if self.entry_profile == PROFILE_WORKFLOW else None
        plan = None
        async for event in self._router.route_stream(
            user_query,
            history=conversation_history,
            locale=self.locale,
            previous_step_info=get_thread_stage_store().get_last_stage_summary(
                self.domain,
                thread_id,
            ),
            force_profile=force_profile,
            tenant_id=self.user_id,
        ):
            if event.get("type") == "router.plan":
                plan = event["_plan_obj"]
                if self.entry_profile == PROFILE_WORKFLOW:
                    plan.profile = PROFILE_WORKFLOW
            yield public_event(event)

        if plan is None:
            raise RuntimeError("RouterOrchestrator 未收到 router.plan")

        exec_mode = (
            MODE_FIXED_GRAPH
            if self.entry_profile == PROFILE_WORKFLOW
            else profile_to_mode(plan.profile)
        )

        with agent_locale_context(self.locale):
            with request_metrics_context(
                domain=self.domain,
                mode=exec_mode,
                transport=self.transport,
            ):
                backend = await self._get_backend(exec_mode)
                if exec_mode == MODE_FIXED_GRAPH:
                    pipeline = self._router_unified_pipeline()
                    prefilled = enrich_execution_plan_pipeline_metadata(
                        ensure_execution_plan_from_routing_plan(
                            plan,
                            user_query=plan.execution_query,
                        ),
                        pre_survey_mode=pipeline.pre_survey_mode,
                    )
                    async for event in backend.iter_request_stream(
                        plan.execution_query,
                        thread_id=thread_id,
                        prefilled_execution_plan=prefilled,
                    ):
                        if event.get("type") == "final":
                            payload = dict(event.get("data") or {})
                            payload["routing_plan"] = plan.to_dict()
                            payload["profile"] = self.entry_profile
                            payload["resolved_profile"] = plan.profile
                            payload["orchestration_mode"] = exec_mode
                            enrich_routing_observability(payload, domain=self.domain)
                            yield final_event(payload)
                        else:
                            yield public_event(event)
                    return

                backend_kwargs: Dict[str, Any] = {}
                async for event in backend.iter_request_stream(
                    plan.execution_query,
                    thread_id=thread_id,
                    timeout_sec=timeout_sec,
                    **backend_kwargs,
                ):
                    if event.get("type") == "final":
                        payload = dict(event.get("data") or {})
                        payload["routing_plan"] = plan.to_dict()
                        payload["profile"] = self.entry_profile
                        payload["resolved_profile"] = plan.profile
                        payload["orchestration_mode"] = exec_mode
                        enrich_routing_observability(payload, domain=self.domain)
                        yield final_event(payload)
                    else:
                        yield public_event(event)
