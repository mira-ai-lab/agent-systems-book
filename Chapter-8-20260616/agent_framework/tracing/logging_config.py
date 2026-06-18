"""结构化 logging：与 OpenTelemetry trace 上下文联动。

每条日志自动附带 trace_id / span_id（TraceContextFilter），
便于在日志与 traces/*.jsonl 之间交叉检索同一请求。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from opentelemetry import trace

_logging_configured = False


def get_trace_ids() -> tuple[Optional[str], Optional[str]]:
    """从当前 OTel context 读取 trace_id / span_id（十六进制）。"""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


class TraceContextFilter(logging.Filter):
    """为每条 LogRecord 注入 trace_id / span_id 字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id, span_id = get_trace_ids()
        record.trace_id = trace_id or "-"
        record.span_id = span_id or "-"
        return True


class StructuredTextFormatter(logging.Formatter):
    """key=value 文本格式，便于 grep 与日志采集平台解析。"""

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"time={datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"trace_id={getattr(record, 'trace_id', '-')}",
            f"span_id={getattr(record, 'span_id', '-')}",
            f"msg={record.getMessage()}",
        ]
        # log_info(..., **fields) 通过 extra_fields 传入的业务字段
        for key, value in sorted(getattr(record, "extra_fields", {}).items()):
            parts.append(f"{key}={value}")
        if record.exc_info:
            parts.append(f"exception={self.formatException(record.exc_info)}")
        return " ".join(parts)


class JsonLogFormatter(logging.Formatter):
    """JSON Lines 友好格式；LOG_JSON=true 时启用。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "span_id": getattr(record, "span_id", "-"),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level: Optional[str] = None) -> None:
    """配置根 logger（幂等）；默认 StructuredTextFormatter。"""
    global _logging_configured
    if _logging_configured:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    use_json = os.getenv("LOG_JSON", "false").lower() in ("1", "true", "yes")

    handler = logging.StreamHandler()
    handler.addFilter(TraceContextFilter())
    handler.setFormatter(JsonLogFormatter() if use_json else StructuredTextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # 降低 HTTP 客户端噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_info(logger: logging.Logger, message: str, **fields: Any) -> None:
    """结构化 info 日志：message + 任意 key=value 业务字段。"""
    logger.info(message, extra={"extra_fields": {k: v for k, v in fields.items() if v is not None}})
