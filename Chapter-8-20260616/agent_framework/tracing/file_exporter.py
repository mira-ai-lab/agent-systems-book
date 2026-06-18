"""将 OpenTelemetry span 导出为本地 JSON Lines 文件。

每行一个 span 的完整快照（含 attributes、events、parent 关系），
便于离线分析 trace 树或与日志 trace_id 关联。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import AttributeValue


def _format_ns(ns: int) -> str:
    """OTel 纳秒时间戳 → ISO8601 UTC 字符串。"""
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat()


def resolve_spans_output_file(output_dir: Path, *, filename: str | None = None) -> Path:
    """根据环境变量决定 span 输出文件路径。

    优先级：显式 filename 参数 > OTEL_TRACES_FILENAME > OTEL_TRACES_FILE_MODE
    OTEL_TRACES_FILE_MODE:
      - timestamp（默认）：每次进程启动一个文件，如 spans_20260612_143052.jsonl
      - append / legacy：追加到 spans.jsonl
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        return output_dir / filename

    explicit = (os.getenv("OTEL_TRACES_FILENAME") or "").strip()
    if explicit:
        return output_dir / explicit

    mode = (os.getenv("OTEL_TRACES_FILE_MODE") or "timestamp").strip().lower()
    if mode in ("append", "legacy", "single"):
        return output_dir / "spans.jsonl"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"spans_{stamp}.jsonl"


def _serialize_value(value: AttributeValue) -> object:
    """将 OTel AttributeValue 转为 JSON 可序列化类型。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return str(value)


def _span_to_dict(span) -> dict:
    """ReadableSpan → 扁平 dict，供 json.dumps 写入 jsonl。"""
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
    """线程安全的 JSONL span 导出器；由 BatchSpanProcessor 批量调用 export()。"""

    def __init__(self, output_dir: Path, *, filename: str | None = None) -> None:
        self._output_dir = Path(output_dir)
        self._output_file = resolve_spans_output_file(self._output_dir, filename=filename)
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
