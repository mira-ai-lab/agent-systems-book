"""Phase 7A/7B：OrchestrationBackend 协议与 Supervisor 后端。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.bootstrap.platform import create_runtime
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR
from agent_framework.orchestration.supervisor.agent_names import registry_agent_to_node_name
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator


def test_registry_agent_to_node_name():
    assert registry_agent_to_node_name("WeatherAgent") == "weather_agent"
    assert registry_agent_to_node_name("FAQAgent") == "faq_agent"
    assert registry_agent_to_node_name("EchoAgent") == "echo_agent"


def test_list_domains_includes_modes():
    from agent_framework.domain.plugin_registry import list_domains

    travel = next(d for d in list_domains() if d["name"] == "travel")
    assert MODE_FIXED_GRAPH in travel["modes"]
    assert MODE_SUPERVISOR in travel["modes"]


def test_create_runtime_fixed_graph(monkeypatch):
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
    runtime = create_runtime("demo", mode=MODE_FIXED_GRAPH, enable_memory=False, llm=mock_llm)
    assert isinstance(runtime, RouterOrchestrator)
    assert runtime.entry_profile == "workflow"
    assert runtime.domain == "demo"


def test_create_runtime_supervisor_demo(monkeypatch):
    from langchain_core.messages import AIMessage

    mock_app = MagicMock()
    mock_app.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="echo ok", name="echo_agent")]}
    )
    mock_llm = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.build_supervisor_app",
        lambda *a, **kw: mock_app,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.create_long_term_memory",
        lambda *a, **kw: (None, None),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.load_project_dotenv",
        lambda: None,
    )

    runtime = create_runtime("demo", mode=MODE_SUPERVISOR, enable_memory=False, llm=mock_llm)
    assert isinstance(runtime, SupervisorOrchestrator)
    assert runtime.mode == MODE_SUPERVISOR

    import asyncio

    result = asyncio.run(runtime.process_request("hello", thread_id="t-sup"))
    assert "echo ok" in result["final_response"]
    assert result["orchestration_mode"] == MODE_SUPERVISOR


def test_unsupported_mode_raises():
    from agent_framework.domain.domain_config import DomainConfig
    from agent_framework.domain.domain_prompts import DomainPrompts
    from agent_framework.domain.plugin import DomainPlugin
    from agent_framework.domain.agent_registry import SubAgentRegistry
    from agent_framework.domain.plugin_registry import clear_domains, register_domain

    clear_domains()

    plugin = DomainPlugin(
        name="fixed_only",
        create_registry=lambda: SubAgentRegistry(),
        create_prompts=lambda: DomainPrompts(
            central_agent_system="x",
            aggregation="a",
            facts_prompt="f",
            decomposition_prompt="d",
            dependency_system="ds",
            dependency_user="du",
            agent_routing="ar",
        ),
        create_domain_config=lambda **_: DomainConfig(),
        supported_modes=(MODE_FIXED_GRAPH,),
    )
    register_domain(plugin)
    with pytest.raises(ValueError, match="不支持 mode"):
        create_runtime("fixed_only", mode=MODE_SUPERVISOR, enable_memory=False, llm=MagicMock())

    clear_domains()
    from agent_framework.domain.plugin_registry import ensure_domains_loaded

    ensure_domains_loaded()
