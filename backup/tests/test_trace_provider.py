"""trace_provider 单元测试。"""

from __future__ import annotations

import asyncio

import pytest

from agent_framework.tracing import (
    current_trace_add_event,
    get_current_span_context,
    setup_observability,
    span,
    trace_span,
)
from agent_framework.tracing.trace_provider import (
    LATC_PREFIX,
    LatcPrefixSampler,
    serialize_for_trace,
    span_name,
)


@pytest.fixture(autouse=True)
def _enable_sample_all(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLE_ALL", "1")


def test_setup_observability_idempotent():
    setup_observability()
    setup_observability()


def test_latc_prefix_sampler():
    sampler = LatcPrefixSampler(sample_all=False)
    latc = sampler.should_sample(None, 1, span_name("request"))
    other = sampler.should_sample(None, 1, "travel.request")
    assert latc.decision.name == "RECORD_AND_SAMPLE"
    assert other.decision.name == "DROP"


def test_span_name_uses_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-custom-agent")
    assert span_name("request") == "latc.my-custom-agent.request"


def test_serialize_state_whitelist():
    state = {
        "thread_id": "t1",
        "user_query": "x" * 300,
        "execution_plan": {"large": True},
    }
    out = serialize_for_trace(state, param_name="state")
    assert out["thread_id"] == "t1"
    assert len(out["user_query"]) <= 203
    assert "execution_plan" not in out


def test_serialize_task():
    task = {
        "task_id": "T1",
        "agent": "WeatherAgent",
        "description": "查天气",
        "depends_on": [],
    }
    out = serialize_for_trace(task, param_name="task")
    assert out["task.id"] == "T1"
    assert out["task.agent"] == "WeatherAgent"


@trace_span(name="latc.test.async_fn", attrs_args=["value"])
async def _sample_async(value: str) -> dict:
    current_trace_add_event("custom.event", {"k": "v"})
    return {"value": value}


def test_trace_span_async_decorator():
    setup_observability()
    result = asyncio.run(_sample_async("hello"))
    assert result["value"] == "hello"


def test_trace_span_async_with_context():
    setup_observability()

    @trace_span(name="latc.test.inner", attrs_args=["msg"])
    async def inner(msg: str) -> str:
        tid, sid = get_current_span_context()
        assert tid
        assert sid
        return msg

    assert asyncio.run(inner("ok")) == "ok"


def test_span_compat_layer():
    setup_observability()
    with span("latc.test.compat", step="unit_test"):
        trace_id, span_id = get_current_span_context()
        assert trace_id
        assert span_id


def test_span_exception_propagates():
    setup_observability()
    with pytest.raises(ValueError, match="boom"):
        with span("latc.test.error", step="unit_test"):
            raise ValueError("boom")


def test_trace_span_name_warning_logged(caplog):
    setup_observability()

    @trace_span(name="legacy.span.no_prefix")
    def legacy():
        return 1

    assert legacy() == 1


def test_latc_prefix_constant():
    assert LATC_PREFIX == "latc."
