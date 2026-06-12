"""Tracing 与结构化 logging 基础测试。"""

import json

import pytest

from travel_multi_agent.tracing import get_current_span_context, setup_observability, span


@pytest.fixture(autouse=True)
def _enable_sample_all(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLE_ALL", "1")


def test_setup_observability_idempotent():
    setup_observability()
    setup_observability()


def test_span_sets_trace_context():
    setup_observability()
    with span("latc.test.span", step="unit_test"):
        trace_id, span_id = get_current_span_context()
        assert trace_id
        assert span_id


def test_span_exception_with_duplicate_step_attribute():
    setup_observability()
    with pytest.raises(ValueError, match="boom"):
        with span("latc.test.execute_layer", step="execute_layer.1"):
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
        name="latc.test.file_export",
        kind=SpanKind.INTERNAL,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        status=status,
        attributes={"step": "unit_test"},
        events=[],
    )

    exporter = FileSpanExporter(tmp_path, filename="spans_test.jsonl")
    exporter.export([fake_span])

    spans_file = tmp_path / "spans_test.jsonl"
    assert spans_file.is_file()
    record = json.loads(spans_file.read_text(encoding="utf-8").strip())
    assert record["trace_id"] == format(ctx.trace_id, "032x")
    assert record["name"] == "latc.test.file_export"
    assert record["attributes"]["step"] == "unit_test"


def test_file_exporter_timestamp_mode(tmp_path, monkeypatch):
    from travel_multi_agent.tracing.file_exporter import FileSpanExporter, resolve_spans_output_file

    monkeypatch.setenv("OTEL_TRACES_FILE_MODE", "timestamp")
    monkeypatch.delenv("OTEL_TRACES_FILENAME", raising=False)
    path = resolve_spans_output_file(tmp_path)
    assert path.name.startswith("spans_")
    assert path.suffix == ".jsonl"
    assert path.name != "spans.jsonl"

    exporter = FileSpanExporter(tmp_path)
    assert exporter.output_file == path
