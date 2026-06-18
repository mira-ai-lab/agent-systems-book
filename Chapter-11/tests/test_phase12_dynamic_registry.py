"""Phase 12：动态 Agent Registry。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent_framework.bootstrap.tenant_pool import TenantOrchestratorPool
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.dynamic_registry import (
    DynamicAgentRecord,
    DynamicAgentStore,
    get_dynamic_agent_store,
    merge_dynamic_agents,
    reset_dynamic_agent_store,
)
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine


@pytest.fixture(autouse=True)
def _reset_dynamic_store():
    reset_dynamic_agent_store()
    yield
    reset_dynamic_agent_store()


def test_sub_agent_registry_metadata_only():
    registry = SubAgentRegistry()
    registry.register_metadata("RemoteAgent", description="远程 Agent")
    assert registry.has_agent("RemoteAgent")
    assert registry.is_metadata_only("RemoteAgent")
    try:
        registry.get_agent("RemoteAgent")
        assert False, "metadata-only agent should not instantiate"
    except ValueError as exc:
        assert "元数据" in str(exc)


def test_dynamic_store_register_and_unregister():
    store = DynamicAgentStore()
    record = store.register(
        "demo",
        DynamicAgentRecord(name="BillingAgent", description="账单查询"),
    )
    assert record.name == "BillingAgent"
    agents = store.list_agents("demo")
    assert len(agents) == 1
    assert store.unregister("demo", "BillingAgent") is True
    assert store.list_agents("demo") == []


def test_merge_dynamic_agents_into_registry():
    reset_dynamic_agent_store()
    base = SubAgentRegistry()
    base.register("EchoAgent", lambda: object(), description="echo")
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(
            name="RemoteFAQ",
            description="远程 FAQ",
            source="a2a",
            a2a_url="http://127.0.0.1:9100/",
        ),
    )
    merged, a2a = merge_dynamic_agents("demo", base)
    names = merged.get_agent_names()
    assert "EchoAgent" in names
    assert "RemoteFAQ" in names
    assert merged.is_metadata_only("RemoteFAQ")
    assert len(a2a) == 1
    assert a2a[0].url.startswith("http://")


def test_router_engine_sees_dynamic_agent():
    reset_dynamic_agent_store()
    plugin = get_domain_plugin("demo")
    base = plugin.create_registry()
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="PolicyAgent", description="政策库检索"),
    )
    merged, _ = merge_dynamic_agents("demo", base)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["政策库检索"]'),
            AIMessage(content='[{"name": "PolicyAgent", "score": 0.95}]'),
        ]
    )
    plan = asyncio.run(
        RouterEngine(
            mock_llm,
            merged,
            config=RouterConfig(enable_instruction_build=False),
        ).route("查询报销政策")
    )
    assert plan.candidates[0].name == "PolicyAgent"


def test_tenant_pool_invalidate_domain():
    pool = TenantOrchestratorPool(max_size=4)
    pool._cache["demo:auto:router:local:u1"] = object()  # type: ignore[assignment]
    pool._cache["travel:workflow:fixed_graph:local:u1"] = object()  # type: ignore[assignment]
    removed = pool.invalidate("demo")
    assert removed == 1
    assert "travel:workflow:fixed_graph:local:u1" in pool._cache


def test_persisted_dynamic_agent_store(tmp_path, monkeypatch):
    from agent_framework.domain.dynamic_registry import get_dynamic_agent_store, reset_dynamic_agent_store
    from agent_framework.domain.dynamic_registry_persist import PersistedDynamicAgentStore

    db_path = tmp_path / "agents.json"
    monkeypatch.setenv("DYNAMIC_AGENTS_PATH", str(db_path))
    reset_dynamic_agent_store()

    store_a = PersistedDynamicAgentStore(db_path)
    store_a.register("demo", DynamicAgentRecord(name="PersistAgent", description="持久化测试"))
    assert db_path.is_file()

    store_b = PersistedDynamicAgentStore(db_path)
    agents = store_b.list_agents("demo")
    assert len(agents) == 1
    assert agents[0].name == "PersistAgent"

    reset_dynamic_agent_store()
