"""Tracing 与结构化 logging 基础测试。"""

import json

import pytest

from travel_multi_agent.tracing import get_trace_ids, setup_observability, span


def test_setup_observability_idempotent():
    setup_observability()
    setup_observability()


def test_span_sets_trace_context():
    setup_observability()
    with span("test.span", step="unit_test"):
        trace_id, span_id = get_trace_ids()
        assert trace_id
        assert span_id


def test_span_exception_with_duplicate_step_attribute():
    setup_observability()
    with pytest.raises(ValueError, match="boom"):
        with span("orchestration.execute_layer.1", step="execute_layer.1"):
            raise ValueError("boom")


def test_file_exporter_writes_spans(tmp_path):
    from types import SimpleNamespace

    from opentelemetry.trace import SpanKind, StatusCode

    from travel_multi_agent.tracing.file_exporter import FileSpanExporter

    ctx = SimpleNamespace(trace_id=0xABC, span_id=0x123)
    status = SimpleNamespace(status_code=StatusCode.OK, description="")
    fake_span = SimpleNamespace(
        get_span_context=lambda: ctx,
        parent=None,
        name="test.file_export",
        kind=SpanKind.INTERNAL,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        status=status,
        attributes={"step": "unit_test"},
        events=[],
    )

    exporter = FileSpanExporter(tmp_path)
    exporter.export([fake_span])

    spans_file = tmp_path / "spans.jsonl"
    assert spans_file.is_file()
    record = json.loads(spans_file.read_text(encoding="utf-8").strip())
    assert record["trace_id"] == format(ctx.trace_id, "032x")
    assert record["name"] == "test.file_export"
    assert record["attributes"]["step"] == "unit_test"
