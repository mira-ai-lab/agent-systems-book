"""Phase 15：thread stage 累积 + knowledge 路由 + Sample Domain。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.plugin_registry import list_domains
from agent_framework.orchestration.thread_stage_context import (
    PersistedThreadStageContextStore,
    reset_thread_stage_store,
)
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine
from agent_framework.router.skills_format import format_agent_skills
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.stages.knowledge_routing import (
    match_knowledge_candidates,
    merge_agent_candidates,
)


@pytest.fixture(autouse=True)
def _reset_thread_store():
    reset_thread_stage_store()
    yield
    reset_thread_stage_store()


def test_thread_stage_store_persisted(tmp_path):
    db_path = tmp_path / "thread_stage.json"
    store_a = PersistedThreadStageContextStore(db_path)
    store_a.set_last_stage_summary("demo", "t1", "阶段一已完成退货咨询")
    assert db_path.is_file()

    store_b = PersistedThreadStageContextStore(db_path)
    assert store_b.get_last_stage_summary("demo", "t1") == "阶段一已完成退货咨询"


def test_format_agent_skills_with_knowledge_keyword():
    info = {
        "skills": [
            {
                "name": "知识支持-退换货政策",
                "description": "政策库",
                "tags": ["政策-退货", "政策-换货"],
                "keywords": ["退货"],
            }
        ]
    }
    text = format_agent_skills(info, locale="zh")
    assert "知识支持" in text
    assert "包含的知识点有" in text
    assert "退货" in text


def test_match_knowledge_candidates():
    registry = SubAgentRegistry()
    registry.register_metadata(
        "FAQAgent",
        description="FAQ",
        skills=[
            {
                "name": "知识支持-退换货政策",
                "tags": ["政策-退货"],
                "keywords": ["退货"],
            }
        ],
    )
    hits = match_knowledge_candidates(registry, query="我想咨询退货政策", events=[])
    assert len(hits) == 1
    assert hits[0].name == "FAQAgent"
    assert hits[0].score >= 0.65


def test_merge_agent_candidates_takes_max_score():
    merged = merge_agent_candidates(
        [AgentCandidate("A", 0.6)],
        [AgentCandidate("A", 0.9)],
    )
    assert merged[0].score == 0.9


def test_router_engine_knowledge_boost():
    registry = SubAgentRegistry()
    registry.register_metadata(
        "FAQAgent",
        description="FAQ",
        skills=[{"name": "知识支持-政策", "tags": ["政策-退货"], "keywords": ["退货"]}],
    )
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货政策"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.4}]'),
        ]
    )
    engine = RouterEngine(
        mock_llm,
        registry,
        config=RouterConfig(
            enable_history_gate=False,
            enable_interaction_rewrite=False,
            enable_instruction_build=False,
        ),
    )
    plan = asyncio.run(engine.route("我要退货"))
    faq = next(c for c in plan.candidates if c.name == "FAQAgent")
    assert faq.score > 0.4
    assert faq.score <= 1.0
    assert "knowledge_routing" in plan.metadata["stages"]
    matches = plan.metadata.get("knowledge_matches") or []
    assert any(m.get("normalized_score") is not None for m in matches)


def test_travel_is_product_domain():
    domains = list_domains()
    travel = next(d for d in domains if d["name"] == "travel")
    cs = next(d for d in domains if d["name"] == "customer_service")
    assert travel["is_sample"] is False
    assert cs["is_sample"] is False
    assert travel["recommended"] is True
    assert cs["recommended"] is True
