"""平台级编排运行时工厂：支持 workflow / adaptive / auto 路由。"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI

from agent_framework.domain.dynamic_registry import resolve_domain_registry_and_a2a
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.infra.agent_runtime import configure_agent_llm
from agent_framework.orchestration.protocol import (
    MODE_FIXED_GRAPH,
    MODE_SUPERVISOR,
    TRANSPORT_LOCAL,
    AgentTransport,
    OrchestrationBackend,
    OrchestrationMode,
)
from agent_framework.orchestration.router_orchestrator import RouterOrchestrator
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator
from agent_framework.router.config import RouterConfig
from agent_framework.router.profile import PROFILE_AUTO, PROFILE_HYBRID, PROFILE_WORKFLOW, normalize_profile, profile_to_mode


def _resolve_execution_profile(
    profile: str,
    mode: Optional[OrchestrationMode],
) -> str:
    if mode is not None and (not profile or profile == PROFILE_WORKFLOW):
        if mode == MODE_SUPERVISOR:
            return "adaptive"
        if mode == MODE_FIXED_GRAPH:
            return PROFILE_WORKFLOW
    return normalize_profile(profile or PROFILE_WORKFLOW)


def _normalize_transport(
    transport: Optional[str],
    *,
    mode: OrchestrationMode,
    profile: str = PROFILE_WORKFLOW,
) -> AgentTransport:
    value = (transport or TRANSPORT_LOCAL).strip() or TRANSPORT_LOCAL
    if profile == PROFILE_AUTO:
        if value not in ("local", "a2a", "mixed"):
            raise ValueError("agent_transport 可选: local, a2a, mixed")
        return value  # type: ignore[return-value]
    if profile == PROFILE_HYBRID:
        if (transport or "").strip() == "":
            return "mixed"  # type: ignore[return-value]
        if value not in ("local", "a2a", "mixed"):
            raise ValueError("agent_transport 可选: local, a2a, mixed")
        return value  # type: ignore[return-value]
    if mode != MODE_SUPERVISOR and value != TRANSPORT_LOCAL:
        raise ValueError("agent_transport 仅适用于 mode='supervisor'、profile='auto' 或 profile='hybrid'")
    if value not in ("local", "a2a", "mixed"):
        raise ValueError("agent_transport 可选: local, a2a, mixed")
    return value  # type: ignore[return-value]


def create_runtime(
    domain: str,
    *,
    profile: str = PROFILE_WORKFLOW,
    mode: Optional[OrchestrationMode] = None,
    transport: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    user_id: str = "default",
    enable_memory: bool = True,
    enable_guess_agent: bool = True,
    long_term_backend: Optional[Any] = None,
    pipeline: Optional[PipelineConfig] = None,
    locale: str = "zh",
    router_config: Optional[RouterConfig] = None,
) -> OrchestrationBackend:
    """按领域插件与执行 Profile 创建运行时。"""
    plugin = get_domain_plugin(domain)
    execution_profile = _resolve_execution_profile(profile, mode)

    resolved_llm = llm
    if resolved_llm is None:
        from agent_framework.config import create_llm

        resolved_llm = create_llm()

    configure_agent_llm(resolved_llm)

    if execution_profile == PROFILE_AUTO:
        if not plugin.supports_mode(MODE_FIXED_GRAPH):
            raise ValueError(f"领域 '{domain}' 需要支持 fixed_graph 以供 auto 路由")
        agent_transport = _normalize_transport(
            transport,
            mode=MODE_SUPERVISOR,
            profile=execution_profile,
        )
        return RouterOrchestrator(
            resolved_llm,
            plugin,
            domain=domain,
            user_id=user_id,
            enable_memory=enable_memory,
            enable_guess_agent=enable_guess_agent,
            long_term_backend=long_term_backend,
            transport=agent_transport,
            locale=locale,
            entry_profile=PROFILE_AUTO,
            router_config=router_config,
        )

    if execution_profile == PROFILE_WORKFLOW:
        if not plugin.supports_mode(MODE_FIXED_GRAPH):
            raise ValueError(f"领域 '{domain}' 需要支持 fixed_graph 以供 workflow 路由")
        if transport and (transport or "").strip() not in ("", TRANSPORT_LOCAL):
            raise ValueError(
                "agent_transport 仅适用于 mode='supervisor'、profile='auto' 或 profile='hybrid'"
            )
        return RouterOrchestrator(
            resolved_llm,
            plugin,
            domain=domain,
            user_id=user_id,
            enable_memory=enable_memory,
            enable_guess_agent=enable_guess_agent,
            long_term_backend=long_term_backend,
            transport=TRANSPORT_LOCAL,
            locale=locale,
            entry_profile=PROFILE_WORKFLOW,
            router_config=router_config,
        )

    if execution_profile == PROFILE_HYBRID:
        if not plugin.supports_mode(MODE_SUPERVISOR):
            supported = ", ".join(plugin.supported_modes)
            raise ValueError(
                f"领域 '{domain}' 不支持 hybrid profile，需要 supervisor，可选: {supported}"
            )
        agent_transport = _normalize_transport(
            transport or "mixed",
            mode=MODE_SUPERVISOR,
            profile=execution_profile,
        )
        registry, a2a_endpoints = resolve_domain_registry_and_a2a(domain, plugin)
        prompts = plugin.create_prompts(locale=locale).with_platform_defaults(locale)
        pipe = pipeline or plugin.build_pipeline(enable_memory=enable_memory)
        return SupervisorOrchestrator(
            resolved_llm,
            domain=domain,
            user_id=user_id,
            registry=registry,
            prompts=prompts,
            pipeline=pipe,
            enable_memory=enable_memory,
            long_term_backend=long_term_backend,
            transport=agent_transport,
            a2a_endpoints=a2a_endpoints,
            locale=locale,
        )

    orchestration_mode: OrchestrationMode = mode or profile_to_mode(execution_profile)
    agent_transport = _normalize_transport(
        transport,
        mode=orchestration_mode,
        profile=execution_profile,
    )

    if not plugin.supports_mode(orchestration_mode):
        supported = ", ".join(plugin.supported_modes)
        raise ValueError(f"领域 '{domain}' 不支持 mode='{orchestration_mode}'，可选: {supported}")

    registry, a2a_endpoints = resolve_domain_registry_and_a2a(domain, plugin)
    prompts = plugin.create_prompts(locale=locale).with_platform_defaults(locale)
    pipe = pipeline or plugin.build_pipeline(enable_memory=enable_memory)

    if orchestration_mode == MODE_SUPERVISOR:
        return SupervisorOrchestrator(
            resolved_llm,
            domain=domain,
            user_id=user_id,
            registry=registry,
            prompts=prompts,
            pipeline=pipe,
            enable_memory=enable_memory,
            long_term_backend=long_term_backend,
            transport=agent_transport,
            a2a_endpoints=a2a_endpoints,
            locale=locale,
        )

    raise ValueError(
        f"profile='{execution_profile}' 无可用运行时；"
        "workflow/auto 走 RouterOrchestrator，adaptive/hybrid 走 Supervisor"
    )


def create_orchestrator(
    domain: str,
    *,
    llm: Optional[ChatOpenAI] = None,
    user_id: str = "default",
    enable_memory: bool = True,
    enable_guess_agent: bool = True,
    long_term_backend: Optional[Any] = None,
    pipeline: Optional[PipelineConfig] = None,
) -> RouterOrchestrator:
    """创建 Router 统一入口（``create_runtime(..., profile='workflow')`` 的别名）。"""
    runtime = create_runtime(
        domain,
        profile=PROFILE_WORKFLOW,
        llm=llm,
        user_id=user_id,
        enable_memory=enable_memory,
        enable_guess_agent=enable_guess_agent,
        long_term_backend=long_term_backend,
        pipeline=pipeline,
    )
    if not isinstance(runtime, RouterOrchestrator):
        raise TypeError("internal error: expected RouterOrchestrator")
    return runtime
