"""Phase 20：SSE + Router stage events + FixedGraph 节点进度。"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.router_orchestrator import RouterOrchestrator
from agent_framework.router.engine import RouterEngine
from agent_framework.router.plan import AgentCandidate
from agent_framework.stream.events import public_event
from agent_framework.stream.sse import format_sse


def _cs_registry() -> SubAgentRegistry:
    registry = SubAgentRegistry()
    registry.register("FAQAgent", lambda: MagicMock(), description="FAQ 政策咨询")
    registry.register("TicketAgent", lambda: MagicMock(), description="工单投诉")
    return registry


def test_route_stream_yields_stage_events():
    registry = _cs_registry()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.9}, {"name": "TicketAgent", "score": 0.85}]'),
            AIMessage(content="整体目标：处理退货\n子任务：\n- 查政策\n- 提交工单"),
        ]
    )
    engine = RouterEngine(mock_llm, registry)

    async def collect():
        events = []
        async for event in engine.route_stream("我要咨询退货并投诉"):
            events.append(public_event(event))
        return events

    events = asyncio.run(collect())
    types = [event["type"] for event in events]
    assert "router.extraction" in types
    assert "router.classification" in types
    assert "router.task_decomposition" in types
    assert types[-1] == "router.plan"
    extraction = next(event for event in events if event["type"] == "router.extraction")
    assert extraction["data"]["events"]


def test_format_sse_event():
    payload = format_sse(
        {
            "type": "router.extraction",
            "stage": "extraction",
            "data": {"events": ["咨询退货"]},
        },
        event_id="1",
    )
    assert payload.startswith("id: 1\n")
    assert "event: router.extraction\n" in payload
    assert "data: " in payload
    data_line = next(line for line in payload.splitlines() if line.startswith("data: "))
    body = json.loads(data_line[len("data: ") :])
    assert body["data"]["events"] == ["咨询退货"]


def test_langgraph_iter_request_stream_emits_graph_events(monkeypatch):
    from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator

    mock_app = MagicMock()

    async def fake_astream(*args, **kwargs):
        yield {"pre_survey": {"pre_survey": {"given_facts": ["x"]}, "logs": []}}
        yield {"build_plan": {"execution_plan": {"subtasks": [{"task_id": "T1"}]}, "logs": []}}
        yield {"aggregate": {"final_response": "done", "logs": []}}

    mock_app.astream = fake_astream
    mock_app.aget_state = AsyncMock(
        return_value=MagicMock(values={"final_response": "done", "subtask_results": {"T1": {"status": "ok"}}})
    )

    mock_llm = MagicMock()
    orch = LangGraphOrchestrator(
        llm=mock_llm,
        enable_memory=False,
        domain="demo",
    )
    orch.app = mock_app

    async def collect():
        events = []
        async for event in orch.iter_request_stream("hello", thread_id="t1"):
            events.append(event)
        return events

    events = asyncio.run(collect())
    types = [event["type"] for event in events]
    assert "graph.node" in types
    assert types[-1] == "final"
    assert events[-1]["data"]["final_response"] == "done"


def test_router_orchestrator_stream_composes_router_and_graph():
    from agent_framework.domain.pipeline import PipelineConfig

    plugin = MagicMock()
    plugin.create_prompts.return_value.with_platform_defaults.return_value = MagicMock()
    plugin.create_domain_config.return_value = MagicMock()
    plugin.build_pipeline.return_value = PipelineConfig(
        enable_pre_survey=False,
        enable_memory=False,
    )
    plugin.supports_mode.return_value = True

    registry = _cs_registry()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["咨询退货"]'),
            AIMessage(content='[{"name": "FAQAgent", "score": 0.95}]'),
            AIMessage(content="整体目标：咨询\n子任务：\n- 查政策"),
        ]
    )

    graph_backend = MagicMock()

    async def fake_graph_stream(*args, **kwargs):
        yield {"type": "graph.node", "stage": "aggregate", "data": {"node": "aggregate"}}
        yield {
            "type": "final",
            "stage": "done",
            "data": {"final_response": "ok", "trace_id": "t", "span_id": "s"},
        }

    graph_backend.iter_request_stream = fake_graph_stream

    orch = RouterOrchestrator(
        mock_llm,
        plugin,
        domain="travel",
        enable_memory=False,
        entry_profile="workflow",
    )
    orch.registry = registry
    orch._router.registry = registry
    orch._get_backend = AsyncMock(return_value=graph_backend)

    with patch(
        "agent_framework.orchestration.router_orchestrator.get_thread_stage_store",
        return_value=MagicMock(get_last_stage_summary=MagicMock(return_value="")),
    ), patch(
        "agent_framework.router.stages.semantic_routing.should_use_semantic_routing",
        return_value=False,
    ):
        events = asyncio.run(_collect_stream(orch, "查北京明天天气"))

    types = [event["type"] for event in events]
    assert "router.extraction" in types
    assert "router.plan" in types
    assert "graph.node" in types
    assert types[-1] == "final"
    assert events[-1]["data"]["routing_plan"]["profile"] == "workflow"
    assert events[-1]["data"]["final_response"] == "ok"


async def _collect_stream(orch, query):
    events = []
    async for event in orch.iter_request_stream(query, thread_id="t-stream"):
        events.append(public_event(event))
    return events
