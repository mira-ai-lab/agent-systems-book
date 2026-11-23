"""Phase 14：selection.extraction 事件抽取。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.router.config import RouterConfig
from agent_framework.router.engine import RouterEngine
from agent_framework.router.stages.extraction import parse_extraction_response, run_extraction


def test_parse_extraction_response_list():
    events = parse_extraction_response('["处理VPN连接问题", "查询报销政策"]')
    assert events == ["处理VPN连接问题", "查询报销政策"]


def test_parse_extraction_response_multi_tuple():
    events = parse_extraction_response('[0, "", ["处理退货申请"]]')
    assert events == ["处理退货申请"]


def test_run_extraction_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='["解决Teams启动失败问题"]'))
    events = asyncio.run(run_extraction(mock_llm, "Teams 打不开错误500", locale="zh"))
    assert events == ["解决Teams启动失败问题"]


def test_router_engine_with_extraction():
    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货政策"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.95}]'),
        ]
    )
    engine = RouterEngine(
        mock_llm,
        registry,
        config=RouterConfig(
            enable_history_gate=False,
            enable_interaction_rewrite=False,
            enable_extraction=True,
            enable_instruction_build=False,
        ),
    )
    plan = asyncio.run(engine.route("我要退货"))
    assert plan.events == ["咨询退货政策"]
    assert "extraction" in plan.metadata["stages"]
