"""Phase 11：history_gate + interaction_rewrite + instruction_build + locale fallback。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.router.config import RouterConfig
from agent_framework.router.context.history import format_history_text, HistoryTurn
from agent_framework.router.engine import RouterEngine
from agent_framework.router.instruction_builder import InstructionBuilder
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.stages.history_gate import parse_history_gate_response
from agent_framework.router.stages.history_gate import run_history_gate
from agent_framework.router.stages.interaction_rewrite import run_interaction_rewrite


def _cs_registry() -> SubAgentRegistry:
    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ 政策咨询")
    registry.register("TicketAgent", lambda: MagicMock(), description="工单投诉")
    return registry


def test_parse_history_gate_response():
    assert parse_history_gate_response("1") is True
    assert parse_history_gate_response("0") is False
    assert parse_history_gate_response(" 1 ") is True


def test_format_history_text_from_turns():
    text = format_history_text(
        [HistoryTurn("退货怎么弄", "请提供订单号"), HistoryTurn("12345", "已记录")]
    )
    assert "第1轮" in text
    assert "退货怎么弄" in text
    assert "12345" in text


def test_domain_prompts_platform_fallback():
    prompts = DomainPrompts(
        central_agent_system="",
        aggregation="",
        facts_prompt="",
        decomposition_prompt="",
        dependency_system="",
        dependency_user="",
        agent_routing="",
        supervisor_system="",
    ).with_platform_defaults("zh")
    assert prompts.central_agent_system.strip()
    assert "{user_input}" in prompts.decomposition_prompt
    assert prompts.aggregation.strip()


def test_history_gate_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="1"))
    relevant = asyncio.run(
        run_history_gate(
            mock_llm,
            "订单号是 12345",
            "用户: 我要退货\n助手: 请提供订单号",
            locale="zh",
        )
    )
    assert relevant is True


def test_interaction_rewrite_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="申请退货，订单号是 12345")
    )
    rewritten = asyncio.run(
        run_interaction_rewrite(
            mock_llm,
            "12345",
            "第1轮:\n用户: 我要退货\n助手: 请提供订单号",
            locale="zh",
        )
    )
    assert "12345" in rewritten


def test_instruction_builder_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="查询退货政策并说明适用条件")
    )
    builder = InstructionBuilder(mock_llm, locale="zh")
    instruction = asyncio.run(
        builder.build(
            init_task="咨询退货政策",
            target_agent="FAQAgent",
            agent_skill="FAQ 政策咨询",
        )
    )
    assert "退货" in instruction


def test_router_engine_full_pipeline_mock():
    registry = _cs_registry()
    responses = [
        AIMessage(content="1"),
        AIMessage(content="申请退货并咨询物流投诉"),
        AIMessage(content='["申请退货", "咨询物流投诉"]'),
        AIMessage(content='[{"name": "FAQAgent", "score": 0.9}]'),
        AIMessage(content="处理退货咨询与物流投诉相关工单"),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=responses)

    config = RouterConfig(
        enable_history_gate=True,
        enable_interaction_rewrite=True,
        enable_instruction_build=True,
        enable_classification=True,
    )
    engine = RouterEngine(mock_llm, registry, config=config)
    history = "第1轮:\n用户: 我要退货\n助手: 请补充具体问题"
    plan = asyncio.run(engine.route("还有物流很慢", history=history, locale="zh"))

    assert plan.history_relevant is True
    assert plan.rewritten_query == "申请退货并咨询物流投诉"
    assert plan.profile == "adaptive"
    assert plan.primary_agent == "FAQAgent"
    assert plan.agent_instruction == "处理退货咨询与物流投诉相关工单"
    assert plan.execution_query == plan.agent_instruction
    assert "history_gate" in plan.metadata["stages"]
    assert "interaction_rewrite" in plan.metadata["stages"]
    assert "instruction_build" in plan.metadata["stages"]
    assert plan.events == ["申请退货", "咨询物流投诉"]


def test_router_engine_skips_history_when_irrelevant():
    registry = _cs_registry()
    responses = [
        AIMessage(content="0"),
        AIMessage(content='["处理VPN连接问题"]'),
        AIMessage(content='[{"name": "FAQAgent", "score": 0.9}]'),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=responses)

    config = RouterConfig(
        enable_history_gate=True,
        enable_interaction_rewrite=True,
        enable_instruction_build=False,
    )
    engine = RouterEngine(mock_llm, registry, config=config)
    plan = asyncio.run(
        engine.route("VPN 连不上", history="旧对话无关内容", locale="zh")
    )

    assert plan.history_relevant is False
    assert plan.rewritten_query == "VPN 连不上"
    assert mock_llm.ainvoke.call_count == 3


def test_routing_plan_to_dict_includes_phase11_fields():
    plan_dict = __import__(
        "agent_framework.router.plan", fromlist=["RoutingPlan"]
    ).RoutingPlan(
        rewritten_query="q",
        candidates=[AgentCandidate("FAQAgent", 0.9)],
        history_relevant=True,
        primary_agent="FAQAgent",
        agent_instruction="instr",
    ).to_dict()
    assert plan_dict["history_relevant"] is True
    assert plan_dict["primary_agent"] == "FAQAgent"
    assert plan_dict["execution_query"] == "instr"
