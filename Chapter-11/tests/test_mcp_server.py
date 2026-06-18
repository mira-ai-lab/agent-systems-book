"""travel_agent_mcp_server MCP 工具单元测试（Mock 编排器）。"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")

_CHAPTER8 = Path(__file__).resolve().parent.parent
_MCP_SCRIPT = _CHAPTER8 / "scripts" / "travel_agent_mcp_server.py"


def _load_mcp_module(monkeypatch):
    """加载 MCP server 脚本，并 Mock LangGraphOrchestrator。"""
    mock_instance = MagicMock()
    mock_instance.process_request = AsyncMock(
        return_value={
            "final_response": "北京明天晴，25°C",
            "execution_plan": {"subtasks": [{"task_id": "T1", "agent": "WeatherAgent"}]},
            "subtask_results": {"T1": {"status": "ok", "agent_summary": "晴"}},
            "trace_id": "trace-mcp-001",
            "span_id": "span-mcp-001",
        }
    )
    mock_cls = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(
        "agent_framework.orchestration.fixed_graph.orchestrator.LangGraphOrchestrator",
        mock_cls,
    )
    monkeypatch.setattr(
        "agent_framework.tracing.setup_observability",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.config.load_project_dotenv",
        lambda: None,
    )

    module_name = "_travel_agent_mcp_server_test"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _MCP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod._orchestrator = None  # 重置惰性单例，确保使用 mock
    return mod


@pytest.fixture
def mcp_mod(monkeypatch):
    return _load_mcp_module(monkeypatch)


def test_new_thread_id_format(mcp_mod):
    tid = mcp_mod._new_thread_id()
    assert tid.startswith("mcp-")
    assert len(tid) == len("mcp-") + 8


def test_ask_travel_agent_returns_final_response(mcp_mod):
    text = asyncio.run(mcp_mod.ask_travel_agent("北京明天天气"))
    assert "晴" in text
    mcp_mod._get_orchestrator().process_request.assert_awaited_once()
    call_kwargs = mcp_mod._get_orchestrator().process_request.call_args
    assert call_kwargs[0][0] == "北京明天天气"
    assert call_kwargs[1]["thread_id"].startswith("mcp-")


def test_ask_travel_agent_honors_thread_id(mcp_mod):
    asyncio.run(mcp_mod.ask_travel_agent("上海天气", thread_id="session-42"))
    call_kwargs = mcp_mod._get_orchestrator().process_request.call_args
    assert call_kwargs[1]["thread_id"] == "session-42"


def test_ask_travel_agent_detailed_structure(mcp_mod):
    result = asyncio.run(
        mcp_mod.ask_travel_agent_detailed("北京明天天气", thread_id="t1")
    )
    assert result["final_response"]
    assert result["thread_id"] == "t1"
    assert result["trace_id"] == "trace-mcp-001"
    assert result["execution_plan"]["subtasks"][0]["agent"] == "WeatherAgent"


def test_ask_travel_agent_empty_response_falls_back_to_json(mcp_mod):
    mcp_mod._get_orchestrator().process_request = AsyncMock(
        return_value={"final_response": "", "extra": "data"}
    )
    text = asyncio.run(mcp_mod.ask_travel_agent("test"))
    assert "extra" in text


def test_mcp_fastmcp_registered_tools(mcp_mod):
    assert mcp_mod.mcp.name == "travel-example"
    assert callable(mcp_mod.ask_travel_agent)
    assert callable(mcp_mod.ask_travel_agent_detailed)
