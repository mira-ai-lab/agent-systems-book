"""Phase 21–23：向量 KB + Agent Catalog + domain locales JSON。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_catalog import list_platform_agent_entries, summarize_domain_agents
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.dynamic_registry import (
    SHARED_DOMAIN,
    DynamicAgentRecord,
    get_dynamic_agent_store,
    merge_dynamic_agents,
    reset_dynamic_agent_store,
)
from agent_framework.domain.locale_loader import domain_prompts_from_locale
from agent_framework.domain.plugin_registry import get_domain_plugin
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine
from agent_framework.router.kb.loader import get_domain_knowledge_store, reset_domain_knowledge_cache
from agent_framework.router.stages.knowledge_routing import resolve_knowledge_candidates


@pytest.fixture(autouse=True)
def _reset_stores():
    reset_dynamic_agent_store()
    reset_domain_knowledge_cache()
    yield
    reset_dynamic_agent_store()
    reset_domain_knowledge_cache()


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    reset_domain_knowledge_cache()
    yield tmp_path
    reset_domain_knowledge_cache()


def test_vector_knowledge_store_matches_faq(isolated_kb):
    store = get_domain_knowledge_store("customer_service")
    assert store is not None
    hits = store.match_agents("退换货政策", [], min_score=0.15)
    assert hits
    assert hits[0][0] == "FAQAgent"


def test_resolve_knowledge_hybrid_vector_and_keyword(isolated_kb):
    registry = get_domain_plugin("customer_service").create_registry()
    candidates, meta = resolve_knowledge_candidates(
        registry,
        domain="customer_service",
        query="咨询退货政策",
        events=[],
        config=RouterConfig(knowledge_backend="hybrid", knowledge_min_score=0.15),
    )
    assert candidates
    assert any(item["source"] == "vector" for item in meta)


def test_router_engine_vector_knowledge_metadata(isolated_kb):
    registry = get_domain_plugin("customer_service").create_registry()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.4}]'),
        ]
    )
    plan = asyncio.run(
        RouterEngine(
            mock_llm,
            registry,
            domain="customer_service",
            config=RouterConfig(
                enable_instruction_build=False,
                enable_task_decomposition=False,
                knowledge_backend="vector",
                knowledge_vector_min_score=0.15,
            ),
        ).route("退换货政策")
    )
    assert plan.candidates[0].name == "FAQAgent"
    assert plan.candidates[0].score >= 0.15
    assert any(m.get("source") == "vector" for m in plan.metadata["knowledge_matches"])


def test_shared_dynamic_agent_visible_in_all_domains():
    base = SubAgentRegistry()
    base.register("EchoAgent", lambda: object(), description="echo")
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(
            name="GlobalPolicyAgent",
            description="跨域政策助手",
            scope="shared",
            skills=[{"name": "policy", "keywords": ["政策"]}],
        ),
    )
    merged_demo, _ = merge_dynamic_agents("demo", base)
    merged_cs, _ = merge_dynamic_agents(
        "customer_service",
        get_domain_plugin("customer_service").create_registry(),
    )
    assert "GlobalPolicyAgent" in merged_demo.get_agent_names()
    assert "GlobalPolicyAgent" in merged_cs.get_agent_names()
    assert get_dynamic_agent_store().list_agents(SHARED_DOMAIN)[0].name == "GlobalPolicyAgent"


def test_list_platform_agent_entries_includes_static_and_shared():
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="SharedHelper", description="共享助手", scope="shared"),
    )
    entries = list_platform_agent_entries()
    names = {(e["domain"], e["name"]) for e in entries}
    assert ("demo", "EchoAgent") in names
    assert (SHARED_DOMAIN, "SharedHelper") in names


def test_summarize_domain_agents_includes_dynamic():
    get_dynamic_agent_store().register(
        "demo",
        DynamicAgentRecord(name="RuntimeAgent", description="运行时 Agent"),
    )
    text = summarize_domain_agents("demo")
    assert "RuntimeAgent" in text


def test_domain_locale_json_customer_service_zh():
    prompts = domain_prompts_from_locale("customer_service", "zh")
    assert "客服" in prompts.central_agent_system


def test_domain_locale_json_travel_en():
    prompts = domain_prompts_from_locale("travel", "en")
    assert "travel" in prompts.central_agent_system.lower()


def test_demo_prompt_bundle_from_locale():
    from domains.demo.prompt_bundle import DemoPrompts

    prompts = DemoPrompts.build("en")
    assert "Demo orchestrator" in prompts.central_agent_system
