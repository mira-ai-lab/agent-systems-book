"""LangGraphOrchestrator 单元测试（Mock 图与 LLM，不调用真实 API）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.orchestration.fixed_graph.orchestrator import LangGraphOrchestrator
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


@pytest.fixture
def orchestrator(monkeypatch):
    mock_app = MagicMock()
    mock_app.ainvoke = AsyncMock(
        return_value={
            "user_query": "北京天气",
            "final_response": "明天晴，25°C",
            "execution_plan": {"subtasks": [{"task_id": "T1"}]},
            "subtask_results": {"T1": {"status": "ok"}},
            "logs": ["done"],
        }
    )
    async def _empty_astream(*_a, **_kw):
        return
        yield  # pragma: no cover

    mock_app.astream = _empty_astream
    mock_app.aget_state = AsyncMock(
        return_value=MagicMock(
            values={
                "final_response": "明天晴",
                "subtask_results": {},
                "logs": [],
            }
        )
    )

    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.compile_graph",
        lambda *a, **kw: mock_app,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.create_llm",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.create_long_term_memory",
        lambda *a, **kw: (None, None),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.load_project_dotenv",
        lambda: None,
    )

    orch = LangGraphOrchestrator(
        enable_memory=False,
        enable_guess_agent=True,
        registry=create_travel_registry_stub(),
        prompts=TravelPrompts.build(),
        domain_config=DomainConfig(),
    )
    orch.app = mock_app
    return orch


def test_build_initial_state(orchestrator: LangGraphOrchestrator):
    state = orchestrator._build_initial_state("查天气", "thread-1", enable_stream=False)
    assert state["user_query"] == "查天气"
    assert state["thread_id"] == "thread-1"
    assert state["enable_stream"] is False
    assert state["current_layer_index"] == 0
    assert state["subtask_results"] == {}


def test_result_from_state():
    final = {
        "execution_plan": {"x": 1},
        "subtask_results": {"T1": {}},
        "final_response": "ok",
        "logs": ["a"],
    }
    out = LangGraphOrchestrator._result_from_state(final, "trace-abc", "span-def")
    assert out["final_response"] == "ok"
    assert out["trace_id"] == "trace-abc"
    assert out["span_id"] == "span-def"
    assert out["graph_state"] is final


def test_process_request(orchestrator: LangGraphOrchestrator):
    result = asyncio.run(
        orchestrator.process_request("北京明天天气", thread_id="t-weather")
    )
    assert "明天晴" in result["final_response"]
    assert result["subtask_results"]["T1"]["status"] == "ok"
    orchestrator.app.ainvoke.assert_awaited_once()
    call_args = orchestrator.app.ainvoke.call_args
    initial_state = call_args[0][0]
    assert initial_state["user_query"] == "北京明天天气"
    assert call_args[0][1]["configurable"]["thread_id"] == "t-weather"


def test_get_graph_mermaid(orchestrator: LangGraphOrchestrator):
    orchestrator.app.get_graph = MagicMock(
        return_value=MagicMock(draw_mermaid=lambda: "graph TD\n  A-->B")
    )
    assert "graph TD" in orchestrator.get_graph_mermaid()


def test_memory_init_failure_does_not_block(monkeypatch):
    mock_app = MagicMock()
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.compile_graph",
        lambda *a, **kw: mock_app,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.create_llm",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.create_long_term_memory",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("chroma down")),
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.load_project_dotenv",
        lambda: None,
    )

    orch = LangGraphOrchestrator(
        enable_memory=True,
        registry=create_travel_registry_stub(),
        prompts=TravelPrompts.build(),
        domain_config=DomainConfig(),
    )
    assert orch.memory_system is None
    assert orch.app is mock_app
