"""Phase 29：graph.subtask.completed 流式子任务摘要。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.orchestration.fixed_graph.nodes import GraphContext, make_nodes
from agent_framework.orchestration.fixed_graph.stream_sink import StreamSink
from agent_framework.stream.events import (
    build_subtask_summary,
    graph_subtask_completed_event,
    graph_subtask_token_event,
    public_event,
)


def test_build_subtask_summary_prefers_agent_summary():
    text = build_subtask_summary(
        {
            "task_id": "T1",
            "agent": "WeatherAgent",
            "agent_summary": "上海未来三天以晴为主",
            "tool_data": {"city": "上海"},
        }
    )
    assert "上海未来三天" in text


def test_build_subtask_summary_falls_back_to_tool_data():
    text = build_subtask_summary(
        {
            "task_id": "T1",
            "agent": "WeatherAgent",
            "agent_summary": "",
            "tool_call_count": 1,
            "tool_data": {
                "city": "上海",
                "forecasts": [{"day": "Mon", "condition": "晴"}],
            },
        }
    )
    assert "WeatherAgent" in text
    assert "1 次工具调用" in text
    assert "forecasts" not in text


def test_build_subtask_summary_multi_tool_calls():
    text = build_subtask_summary(
        {
            "task_id": "T1",
            "agent": "ItineraryAgent",
            "agent_summary": "",
            "tool_call_count": 6,
            "tool_data": {
                "calls": [
                    {"city": "上海", "candidate_pois": [{}] * 5},
                    {"city": "苏州", "plan": [{}, {}]},
                ],
                "count": 6,
            },
        }
    )
    assert "ItineraryAgent" in text
    assert "6 次工具调用" in text
    assert "candidate_pois" not in text


def test_build_subtask_summary_uses_domain_summarizer():
    def travel_summarizer(result: dict) -> str:
        tool_data = result.get("tool_data") or {}
        city = tool_data.get("city")
        return f"领域摘要：{city}" if city else "领域摘要"

    text = build_subtask_summary(
        {
            "task_id": "T1",
            "agent": "WeatherAgent",
            "agent_summary": "",
            "tool_data": {"city": "上海"},
        },
        summarizer=travel_summarizer,
    )
    assert text == "领域摘要：上海"


def test_graph_subtask_completed_event_schema():
    event = graph_subtask_completed_event(
        {
            "task_id": "T2",
            "agent": "HotelAgent",
            "status": "completed",
            "agent_summary": "推荐静安香格里拉",
        }
    )
    assert event["type"] == "graph.subtask.completed"
    assert event["stage"] == "execute_layer"
    assert event["data"]["task_id"] == "T2"
    assert event["data"]["agent"] == "HotelAgent"
    assert "静安" in event["data"]["summary"]


def test_graph_subtask_token_event_schema():
    event = graph_subtask_token_event("T1", "WeatherAgent", "晴")
    assert event["type"] == "graph.subtask.token"
    assert event["stage"] == "execute_layer"
    assert event["data"] == {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "token": "晴",
    }


def test_stream_sink_emit_subtask_token():
    sink = StreamSink()
    captured: list[tuple[str, str, str]] = []
    sink.enabled = True
    sink.on_subtask_token = lambda task_id, agent, token: captured.append((task_id, agent, token))
    sink.emit_subtask_token("T1", "WeatherAgent", "多")
    sink.emit_subtask_token("T1", "WeatherAgent", "云")
    assert captured == [("T1", "WeatherAgent", "多"), ("T1", "WeatherAgent", "云")]
    sink.reset()
    sink.emit_subtask_token("T1", "WeatherAgent", "x")
    assert len(captured) == 2


def test_stream_sink_emit_subtask_completed():
    sink = StreamSink()
    captured: list[dict] = []
    sink.enabled = True
    sink.on_subtask_completed = captured.append
    payload = {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "status": "completed",
        "agent_summary": "苏州多云",
    }
    sink.emit_subtask_completed(payload)
    assert captured == [payload]
    sink.reset()
    sink.emit_subtask_completed(payload)
    assert len(captured) == 1


def test_public_event_strips_internal_fields():
    event = graph_subtask_completed_event(
        {"task_id": "T1", "agent": "A", "status": "ok", "agent_summary": "x"}
    )
    event["_plan_obj"] = object()
    public = public_event(event)
    assert "_plan_obj" not in public
    assert public["type"] == "graph.subtask.completed"


@patch("agent_framework.orchestration.fixed_graph.nodes._invoke_sub_agent", new_callable=AsyncMock)
def test_execute_layer_emits_subtask_completed_on_stream(mock_invoke):
    mock_invoke.return_value = {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "status": "completed",
        "agent_summary": "杭州 6/27 小雨",
        "tool_data": {"city": "杭州"},
        "tool_call_count": 1,
    }
    sink = StreamSink()
    captured: list[dict] = []
    sink.enabled = True
    sink.on_subtask_completed = captured.append

    ctx = GraphContext(
        MagicMock(),
        None,
        stream_sink=sink,
        registry=MagicMock(),
        prompts=DomainPrompts(
            central_agent_system="sys",
            aggregation="agg",
            facts_prompt="facts",
            decomposition_prompt="decomp",
            dependency_system="dep sys",
            dependency_user="dep user",
            agent_routing="route",
        ),
        domain_config=DomainConfig(),
    )
    nodes = make_nodes(ctx)
    state = {
        "user_query": "规划行程",
        "thread_id": "t-stream",
        "enable_stream": True,
        "pending_layers": [["T1"]],
        "current_layer_index": 0,
        "subtasks": [
            {
                "task_id": "T1",
                "description": "查杭州天气",
                "agent": "WeatherAgent",
                "depends_on": [],
            }
        ],
        "subtask_results": {},
        "logs": [],
    }
    asyncio.run(nodes["execute_layer"](state))
    assert len(captured) == 1
    assert captured[0]["task_id"] == "T1"
    assert captured[0]["agent_summary"] == "杭州 6/27 小雨"


def test_iter_request_stream_forwards_subtask_events():
    from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator

    mock_app = MagicMock()
    subtask_result = {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "status": "completed",
        "agent_summary": "上海晴",
    }

    async def fake_astream(initial_state, config, stream_mode=None):
        sink = orch.stream_sink
        if sink.enabled:
            sink.emit_progress("  🔄 T1 → WeatherAgent")
            sink.emit_subtask_token("T1", "WeatherAgent", "上")
            sink.emit_subtask_token("T1", "WeatherAgent", "海晴")
            sink.emit_subtask_completed(subtask_result)
            sink.emit_progress("\n📝 聚合结果...")
            sink.emit_token("最终")
        yield {"execute_layer": {"subtask_results": {"T1": subtask_result}, "logs": []}}
        yield {"aggregate": {"final_response": "最终回复", "logs": []}}

    mock_app.astream = fake_astream
    mock_app.aget_state = AsyncMock(
        return_value=MagicMock(
            values={
                "final_response": "最终回复",
                "subtask_results": {"T1": subtask_result},
            }
        )
    )

    orch = LangGraphOrchestrator(llm=MagicMock(), enable_memory=False, domain="demo")
    orch.app = mock_app

    async def collect():
        events = []
        async for event in orch.iter_request_stream("hello", thread_id="t1"):
            events.append(event)
        return events

    events = asyncio.run(collect())
    types = [event["type"] for event in events]
    assert "graph.progress" in types
    assert "graph.subtask.token" in types
    assert "graph.subtask.completed" in types
    token_events = [e for e in events if e["type"] == "graph.subtask.token"]
    assert [e["data"]["token"] for e in token_events] == ["上", "海晴"]
    subtask_event = next(e for e in events if e["type"] == "graph.subtask.completed")
    assert subtask_event["data"]["agent"] == "WeatherAgent"
    assert "上海晴" in subtask_event["data"]["summary"]
    assert "graph.token" in types
    assert types[-1] == "final"


def test_iter_request_stream_yields_subtask_tokens_before_node_update():
    """子 Agent token 应在 LangGraph 节点 update 之前实时 yield（Queue 模式）。"""
    from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator

    mock_app = MagicMock()
    subtask_result = {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "status": "completed",
        "agent_summary": "上海晴",
    }
    token_seen = asyncio.Event()

    async def fake_astream(initial_state, config, stream_mode=None):
        sink = orch.stream_sink
        if sink.enabled:
            sink.emit_subtask_token("T1", "WeatherAgent", "上")
            token_seen.set()
            await asyncio.sleep(0.05)
            sink.emit_subtask_completed(subtask_result)
        yield {"execute_layer": {"subtask_results": {"T1": subtask_result}, "logs": []}}

    mock_app.astream = fake_astream
    mock_app.aget_state = AsyncMock(
        return_value=MagicMock(
            values={
                "final_response": "ok",
                "subtask_results": {"T1": subtask_result},
            }
        )
    )

    orch = LangGraphOrchestrator(llm=MagicMock(), enable_memory=False, domain="demo")
    orch.app = mock_app

    async def collect():
        events = []
        async for event in orch.iter_request_stream("hello", thread_id="t-live"):
            events.append(event)
        return events

    async def run():
        task = asyncio.create_task(collect())
        await asyncio.wait_for(token_seen.wait(), timeout=1.0)
        assert task.done() is False
        events = await task
        token_idx = next(
            i for i, event in enumerate(events) if event["type"] == "graph.subtask.token"
        )
        node_idx = next(
            i
            for i, event in enumerate(events)
            if event["type"] == "graph.node"
            and event["data"].get("node") == "execute_layer"
        )
        assert token_idx < node_idx

    asyncio.run(run())
