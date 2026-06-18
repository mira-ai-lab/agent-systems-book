"""Phase 7D：A2A Server 暴露子 Agent。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.transport.a2a.server.serve import resolve_registry_agent


def test_resolve_registry_agent_by_name():
    registry = SubAgentRegistry()
    registry.register("EchoAgent", lambda: MagicMock(), description="echo")
    name = resolve_registry_agent(registry, registry_agent="EchoAgent")
    assert name == "EchoAgent"


def test_resolve_registry_agent_by_node_name():
    registry = SubAgentRegistry()
    registry.register("EchoAgent", lambda: MagicMock(), description="echo")
    name = resolve_registry_agent(registry, node_name="echo_agent")
    assert name == "EchoAgent"


def test_resolve_registry_agent_unknown_raises():
    registry = SubAgentRegistry()
    with pytest.raises(ValueError, match="未知"):
        resolve_registry_agent(registry, registry_agent="Missing")


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("a2a"),
    reason="a2a-sdk not installed",
)
def test_build_sub_agent_executor_invoke():
    from agent_framework.transport.a2a.server.serve import build_sub_agent_executor

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="hotel ok")]}
    )

    with patch(
        "agent_framework.transport.a2a.server.serve.get_domain_plugin"
    ) as mock_plugin:
        registry = SubAgentRegistry()
        registry.register("HotelAgent", lambda: mock_agent, description="hotel")
        plugin = MagicMock()
        plugin.create_registry.return_value = registry
        mock_plugin.return_value = plugin
        with patch("agent_framework.transport.a2a.server.serve.create_llm", return_value=MagicMock()):
            with patch("agent_framework.transport.a2a.server.serve.configure_agent_llm"):
                executor = build_sub_agent_executor("travel", registry_agent="HotelAgent")

    assert executor.factory_name == "HotelAgent"


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("a2a"),
    reason="a2a-sdk not installed",
)
def test_create_sub_agent_a2a_app_builds_starlette():
    from agent_framework.transport.a2a.server.serve import create_sub_agent_a2a_app

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

    with patch(
        "agent_framework.transport.a2a.server.serve.get_domain_plugin"
    ) as mock_plugin:
        registry = SubAgentRegistry()
        registry.register("EchoAgent", lambda: mock_agent, description="echo")
        plugin = MagicMock()
        plugin.create_registry.return_value = registry
        mock_plugin.return_value = plugin
        with patch("agent_framework.transport.a2a.server.serve.create_llm", return_value=MagicMock()):
            with patch("agent_framework.transport.a2a.server.serve.configure_agent_llm"):
                app = create_sub_agent_a2a_app(
                    "demo",
                    registry_agent="EchoAgent",
                    host="127.0.0.1",
                    port=9013,
                )

    assert app is not None
