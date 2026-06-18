"""领域插件与平台 SDK 入口测试。"""

from unittest.mock import MagicMock, patch

import pytest

from agent_framework.domain.plugin_registry import clear_domains, get_domain_plugin, list_domains
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator


def test_list_builtin_domains():
    names = {d["name"] for d in list_domains()}
    assert "travel" in names
    assert "customer_service" in names
    assert "demo" in names


def test_get_domain_plugin_unknown():
    with pytest.raises(KeyError, match="未知领域"):
        get_domain_plugin("finance_not_registered")


def test_orchestrator_requires_domain_or_full_injection():
    with pytest.raises(ValueError, match="缺少领域配置"):
        LangGraphOrchestrator(enable_memory=False)


def test_orchestrator_rejects_partial_injection():
    from agent_framework.domain.domain_config import DomainConfig
    from domains.travel.specs import create_travel_registry_stub

    with pytest.raises(ValueError, match="必须同时注入"):
        LangGraphOrchestrator(
            enable_memory=False,
            registry=create_travel_registry_stub(),
            domain_config=DomainConfig(),
        )


def test_create_orchestrator_travel(monkeypatch):
    from agent_framework.bootstrap.platform import create_orchestrator
    from agent_framework.orchestration.router_orchestrator import RouterOrchestrator

    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )

    orch = create_orchestrator("travel", enable_memory=False, llm=mock_llm)
    assert isinstance(orch, RouterOrchestrator)
    assert orch.domain == "travel"
    assert orch.entry_profile == "workflow"
    assert orch.registry.get_agent_names()
    assert "WeatherAgent" in orch.registry.get_agent_names()


def test_create_orchestrator_customer_service(monkeypatch):
    from agent_framework.bootstrap.platform import create_orchestrator
    from agent_framework.orchestration.router_orchestrator import RouterOrchestrator

    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.router_orchestrator.load_project_dotenv",
        lambda: None,
    )

    orch = create_orchestrator("customer_service", enable_memory=False, llm=mock_llm)
    assert isinstance(orch, RouterOrchestrator)
    assert orch.domain == "customer_service"
    assert orch.entry_profile == "workflow"
    names = orch.registry.get_agent_names()
    assert "FAQAgent" in names
    assert "TicketAgent" in names


def test_register_custom_domain():
    from agent_framework.domain.domain_config import DomainConfig
    from agent_framework.domain.domain_prompts import DomainPrompts
    from agent_framework.domain.plugin import DomainPlugin
    from agent_framework.domain.agent_registry import SubAgentRegistry

    clear_domains()

    def _registry():
        reg = SubAgentRegistry()
        reg.register("EchoAgent", lambda: None, description="echo")
        return reg

    def _prompts():
        return DomainPrompts(
            central_agent_system="x",
            aggregation="a",
            facts_prompt="f",
            decomposition_prompt="d",
            dependency_system="ds",
            dependency_user="du",
            agent_routing="ar",
        )

    plugin = DomainPlugin(
        name="demo",
        create_registry=_registry,
        create_prompts=_prompts,
        create_domain_config=lambda **_: DomainConfig(),
    )
    from agent_framework.domain.plugin_registry import register_domain

    register_domain(plugin)
    resolved = get_domain_plugin("demo")
    assert resolved.name == "demo"
    assert resolved.create_registry().get_agent_names() == ["EchoAgent"]

    clear_domains()
    from agent_framework.domain.plugin_registry import ensure_domains_loaded

    ensure_domains_loaded()
