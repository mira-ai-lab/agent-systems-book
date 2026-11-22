"""Observability package: structured logging plus OpenTelemetry tracing.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from travel_multi_agent.tracing.logging_config import (
    configure_logging,
    get_logger,
    get_trace_ids,
    log_info,
)
from travel_multi_agent.tracing.setup import (
    configure_tracing,
    get_traces_output_dir,
    get_traces_output_file,
    shutdown_tracing,
)
from travel_multi_agent.tracing.spans import record_exception, record_tool_event, span
from travel_multi_agent.tracing.trace_provider import (
    current_trace_add_event,
    get_current_span_context,
    get_service_name,
    get_span_prefix,
    span_name,
    trace_span,
)


def setup_observability(*, service_name: str | None = None) -> None:
    """初始化 logging + tracing（幂等，编排入口调用一次即可）。"""
    configure_logging()
    configure_tracing(service_name=service_name)


__all__ = [
    "configure_logging",
    "configure_tracing",
    "current_trace_add_event",
    "get_current_span_context",
    "get_service_name",
    "get_span_prefix",
    "get_traces_output_dir",
    "get_traces_output_file",
    "get_logger",
    "get_trace_ids",
    "log_info",
    "record_exception",
    "record_tool_event",
    "setup_observability",
    "shutdown_tracing",
    "span",
    "span_name",
    "trace_span",
]