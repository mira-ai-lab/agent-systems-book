"""Phase 24 P1：Registry 产品化（24.9–24.12）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_catalog import (
    build_domain_catalog,
    filter_platform_agent_entries,
    list_platform_agent_entries,
    summarize_domain_agents,
)
from agent_framework.domain.dynamic_registry import (
    DynamicAgentRecord,
    get_dynamic_agent_store,
    merge_dynamic_agents,
    reset_dynamic_agent_store,
)
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.router.engine import RouterEngine
from agent_framework.router.config import RouterConfig
from agent_framework.stream.events import registry_updated_event


@pytest.fixture(autouse=True)
def _reset_store():
    reset_dynamic_agent_store()
    yield
    reset_dynamic_agent_store()


def test_build_domain_catalog_includes_dynamic_and_shared():
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="RuntimeHelper", description="运行时助手"),
    )
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(
            name="SharedFAQ",
            description="共享 FAQ 别名",
            scope="shared",
            alias_of="FAQAgent",
        ),
    )
    catalog = build_domain_catalog()
    assert "RuntimeHelper" in catalog
    assert "SharedFAQ" in catalog
    assert "alias→FAQAgent" in catalog
    assert "跨域共享 Agent" in catalog


def test_summarize_domain_agents_marks_dynamic():
    get_dynamic_agent_store().register(
        "customer_service",
        DynamicAgentRecord(name="RuntimeFAQ", description="运行时 FAQ"),
    )
    text = summarize_domain_agents("customer_service")
    assert "RuntimeFAQ" in text
    assert "dynamic" in text


def test_shared_alias_inherits_static_skills():
    cs_registry = get_domain_plugin("customer_service").create_registry()
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(
            name="GlobalFAQ",
            description="",
            scope="shared",
            alias_of="FAQAgent",
        ),
    )
    merged, _ = merge_dynamic_agents("demo", get_domain_plugin("demo").create_registry())
    assert "GlobalFAQ" in merged.get_agent_names()
    info = merged.agents["GlobalFAQ"]
    assert info.get("references") == "FAQAgent"
    assert info.get("alias_of") == "FAQAgent"
    faq_skills = cs_registry.agents["FAQAgent"].get("skills") or []
    if faq_skills:
        assert info.get("skills")


def test_list_platform_agent_entries_filters():
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="DynA", description="动态 A"),
    )
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="SharedB", description="共享 B", scope="shared"),
    )
    all_entries = list_platform_agent_entries()
    demo_static = list_platform_agent_entries(domain="demo", source="static")
    demo_all = list_platform_agent_entries(domain="demo")
    shared_only = list_platform_agent_entries(scope="shared")
    dynamic_only = list_platform_agent_entries(source="dynamic")

    assert len(all_entries) >= len(demo_all)
    assert all(e.get("source") == "static" for e in demo_static)
    assert any(e["name"] == "EchoAgent" for e in demo_static)
    assert any(e["name"] == "DynA" for e in demo_all)
    assert any(e["name"] == "SharedB" for e in demo_all)
    assert all(e.get("scope") == "shared" for e in shared_only)
    assert all(e.get("source") == "dynamic" for e in dynamic_only)


def test_filter_platform_agent_entries_combinations():
    entries = [
        {"name": "A", "domain": "demo", "scope": "domain", "source": "static"},
        {"name": "B", "domain": "__shared__", "scope": "shared", "source": "dynamic"},
    ]
    assert len(filter_platform_agent_entries(entries, domain="demo")) == 2
    assert len(filter_platform_agent_entries(entries, scope="shared")) == 1
    assert filter_platform_agent_entries(entries, source="static")[0]["name"] == "A"


def test_registry_updated_event_schema():
    event = registry_updated_event(
        domain="demo",
        action="register",
        agent_name="PolicyAgent",
        scope="shared",
    )
    assert event["type"] == "registry.updated"
    assert event["data"]["action"] == "register"
    assert event["data"]["agent_name"] == "PolicyAgent"


def test_notify_registry_updated_webhook(monkeypatch):
    from agent_framework.domain.registry_events import notify_registry_updated

    calls = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            calls.append((url, json))

    monkeypatch.setenv("REGISTRY_WEBHOOK_URL", "http://127.0.0.1:9999/hook")
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: FakeClient())

    event = asyncio.run(notify_registry_updated(domain="demo", action="register", agent_name="X"))
    assert event["type"] == "registry.updated"
    assert calls
    assert calls[0][0] == "http://127.0.0.1:9999/hook"


def test_api_agents_filter_endpoint():
    pytest.importorskip("fastapi")
    from importlib import import_module

    from fastapi.testclient import TestClient

    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="FilterAgent", description="filter test"),
    )
    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    resp = client.get("/v1/agents", params={"domain": "demo", "source": "dynamic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert all(a["source"] == "dynamic" for a in body["agents"])
    assert body["filters"]["domain"] == "demo"


def test_api_register_agent_returns_registry_event():
    pytest.importorskip("fastapi")
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    resp = client.post(
        "/v1/domains/demo/agents",
        json={
            "name": "AliasFAQ",
            "description": "共享 FAQ 入口",
            "scope": "shared",
            "alias_of": "FAQAgent",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["registry_event"]["type"] == "registry.updated"
    assert body["agent"]["alias_of"] == "FAQAgent"


def test_domain_catalog_visible_to_router_classification():
    get_dynamic_agent_store().register(
        "customer_service",
        DynamicAgentRecord(name="HotlineAgent", description="热线升级"),
    )
    registry = get_domain_plugin("customer_service").create_registry()
    merged, _ = merge_dynamic_agents("customer_service", registry)
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["热线"]'),
            AIMessage(content='[{"name": "HotlineAgent", "score": 0.92}]'),
        ]
    )
    plan = asyncio.run(
        RouterEngine(
            mock_llm,
            merged,
            domain="customer_service",
            config=RouterConfig(enable_instruction_build=False),
        ).route("我要转人工热线")
    )
    assert plan.candidates[0].name == "HotlineAgent"
