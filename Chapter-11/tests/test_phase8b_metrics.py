"""Phase 8B：Prometheus mode / transport / a2a_* 指标。"""

from unittest.mock import MagicMock, patch

import pytest

from agent_framework.observability.metrics import (
    endpoint_label,
    record_a2a_call,
    record_chat,
    record_handoff,
    record_job,
    record_job_outcome,
)
from agent_framework.observability.request_context import request_metrics_context


def test_endpoint_label_strips_path():
    assert endpoint_label("http://127.0.0.1:9012/hotel") == "127.0.0.1:9012"


def test_record_chat_with_mode_transport_labels():
    pytest.importorskip("prometheus_client")
    from agent_framework.observability import metrics as metrics_mod

    mock_counter = MagicMock()
    with patch.object(metrics_mod, "CHAT_REQUESTS", mock_counter):
        record_chat("travel", "200", mode="supervisor", transport="mixed")
    mock_counter.labels.assert_called_once_with(
        domain="travel",
        mode="supervisor",
        transport="mixed",
        status="200",
    )


def test_record_a2a_call_uses_request_context_domain():
    pytest.importorskip("prometheus_client")
    from agent_framework.observability import metrics as metrics_mod

    mock_calls = MagicMock()
    mock_hist = MagicMock()
    with patch.object(metrics_mod, "A2A_CALLS", mock_calls):
        with patch.object(metrics_mod, "A2A_CALL_DURATION", mock_hist):
            with request_metrics_context(domain="travel", mode="supervisor", transport="a2a"):
                record_a2a_call("http://127.0.0.1:9012/", status="success", duration_sec=0.42)
    mock_calls.labels.assert_called_once_with(
        domain="travel",
        endpoint="127.0.0.1:9012",
        status="success",
    )
    mock_hist.labels.assert_called_once_with(domain="travel", endpoint="127.0.0.1:9012")


def test_record_handoff_increments_counter():
    pytest.importorskip("prometheus_client")
    from agent_framework.observability import metrics as metrics_mod

    mock_handoffs = MagicMock()
    with patch.object(metrics_mod, "HANDOFFS", mock_handoffs):
        with request_metrics_context(domain="demo", mode="supervisor", transport="local"):
            record_handoff("echo_agent", "local")
    mock_handoffs.labels.assert_called_once_with(
        domain="demo",
        target="echo_agent",
        transport="local",
    )


def test_metrics_endpoint_exposes_new_series():
    pytest.importorskip("prometheus_client")
    from importlib import import_module

    from fastapi.testclient import TestClient

    from agent_framework.observability.metrics import record_a2a_call, record_handoff

    record_chat("demo", "200", mode="supervisor", transport="mixed")
    record_job("demo", mode="supervisor", transport="mixed")
    record_job_outcome("demo", "succeeded", mode="supervisor", transport="mixed")
    with request_metrics_context(domain="travel", mode="supervisor", transport="a2a"):
        record_a2a_call("http://127.0.0.1:9012/", status="success", duration_sec=0.1)
        record_handoff("hotel_agent", "a2a")

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    body = client.get("/metrics").text
    assert "agent_platform_chat_requests_total" in body
    assert "agent_platform_a2a_calls_total" in body
    assert "agent_platform_handoffs_total" in body
    assert "agent_platform_a2a_call_duration_seconds" in body
    assert "agent_platform_job_outcomes_total" in body
