"""将 OpenTelemetry span 写入本地目录（JSON Lines）。"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import AttributeValue


def _format_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _serialize_value(value: AttributeValue) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return str(value)


def _span_to_dict(span) -> dict:
    ctx = span.get_span_context()
    status = span.status
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": (
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        "name": span.name,
        "kind": SpanKind(span.kind).name,
        "start_time": _format_ns(span.start_time),
        "end_time": _format_ns(span.end_time),
        "duration_ms": round((span.end_time - span.start_time) / 1_000_000, 3),
        "status_code": StatusCode(status.status_code).name,
        "status_message": status.description or "",
        "attributes": {
            str(k): _serialize_value(v) for k, v in (span.attributes or {}).items()
        },
        "events": [
            {
                "name": event.name,
                "time": _format_ns(event.timestamp),
                "attributes": {
                    str(k): _serialize_value(v)
                    for k, v in (event.attributes or {}).items()
                },
            }
            for event in span.events
        ],
    }


class FileSpanExporter(SpanExporter):
    """追加写入 `{output_dir}/spans.jsonl`，每行一个 span JSON。"""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file = self._output_dir / "spans.jsonl"
        self._lock = threading.Lock()

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def output_file(self) -> Path:
        return self._output_file

    def export(self, spans: Sequence) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        lines = [json.dumps(_span_to_dict(span), ensure_ascii=False) for span in spans]
        with self._lock:
            with self._output_file.open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass
