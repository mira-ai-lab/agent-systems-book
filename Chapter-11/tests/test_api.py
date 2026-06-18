"""HTTP API 与健康检查测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    from importlib import import_module

    from fastapi import FastAPI

    api_mod = import_module("services.api.app")
    assert isinstance(api_mod.app, FastAPI)

    mock_orch = MagicMock()
    mock_orch.process_request = AsyncMock(
        return_value={
            "final_response": "北京明天晴",
            "trace_id": "trace-test",
            "span_id": "span-test",
        }
    )

    async def default_stream(*args, **kwargs):
        yield {
            "type": "final",
            "stage": "done",
            "data": {
                "final_response": "北京明天晴",
                "trace_id": "trace-test",
                "span_id": "span-test",
            },
        }

    mock_orch.iter_request_stream = default_stream
    with patch.object(api_mod, "_get_orchestrator", AsyncMock(return_value=mock_orch)):
        yield TestClient(api_mod.app), mock_orch


def test_health(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat(api_client):
    client, mock_orch = api_client
    resp = client.post(
        "/v1/chat",
        json={"query": "北京明天天气", "user_id": "alice", "domain": "travel"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert "晴" in body["final_response"]
    assert body["trace_id"] == "trace-test"


def test_chat_auto_domain(api_client):
    client, mock_orch = api_client
    from importlib import import_module

    api_mod = import_module("services.api.app")
    with patch.object(
        api_mod,
        "_resolve_chat_domain",
        AsyncMock(return_value=("demo", [{"name": "demo", "score": 0.95}])),
    ):
        resp = client.post("/v1/chat", json={"query": "hello", "user_id": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["domain"] == "demo"
    assert body["resolved_domain"] == "demo"
    assert body["domain_candidates"][0]["name"] == "demo"


def test_chat_with_locale(api_client):
    client, mock_orch = api_client
    from importlib import import_module

    api_mod = import_module("services.api.app")
    with patch.object(
        api_mod,
        "_resolve_chat_domain",
        AsyncMock(return_value=("demo", None)),
    ):
        resp = client.post(
            "/v1/chat",
            json={"query": "hello", "domain": "demo", "locale": "en"},
        )
    assert resp.status_code == 200
    assert resp.json()["locale"] == "en"


def test_chat_stream_sse(api_client):
    client, mock_orch = api_client

    async def fake_stream(*args, **kwargs):
        yield {
            "type": "router.extraction",
            "stage": "extraction",
            "data": {"events": ["查天气"]},
        }
        yield {
            "type": "final",
            "stage": "done",
            "data": {
                "final_response": "北京明天晴",
                "trace_id": "trace-test",
                "span_id": "span-test",
            },
        }

    mock_orch.iter_request_stream = fake_stream

    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"query": "北京明天天气", "domain": "travel"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.read().decode("utf-8")

    assert "event: router.extraction" in body
    assert "event: final" in body
    assert "北京明天晴" in body


def test_configure_agent_llm(monkeypatch):
    from agent_framework.infra.agent_runtime import build_agent, configure_agent_llm, reset_agent_llm

    sentinel = object()
    configure_agent_llm(sentinel)
    captured = {}

    def fake_create_agent(llm, tools, system_prompt, checkpointer):
        captured["llm"] = llm
        return MagicMock()

    monkeypatch.setattr("agent_framework.infra.agent_runtime.create_agent", fake_create_agent)
    monkeypatch.setattr("agent_framework.infra.agent_runtime.MemorySaver", lambda: MagicMock())

    build_agent([], "sys")
    assert captured["llm"] is sentinel
    reset_agent_llm()
