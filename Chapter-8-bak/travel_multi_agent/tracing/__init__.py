"""
可观测性：结构化 logging + OpenTelemetry tracing。

环境变量：
  OTEL_SERVICE_NAME          服务名（默认 travel-multi-agent）
  OTEL_TRACES_EXPORTER       console | file | otlp | none（默认 console）
  OTEL_TRACES_DIR            file 模式输出目录（默认 Chapter-8/traces/）
  OTEL_EXPORTER_OTLP_ENDPOINT OTLP 地址（默认 http://localhost:4317）
  LOG_LEVEL                  日志级别（默认 INFO）
  LOG_JSON                   true 时输出 JSON 行日志
"""

from travel_multi_agent.tracing.logging_config import (
    configure_logging,
    get_logger,
    get_trace_ids,
    log_info,
)
from travel_multi_agent.tracing.setup import (
    configure_tracing,
    get_traces_output_dir,
    shutdown_tracing,
)
from travel_multi_agent.tracing.spans import record_exception, record_tool_event, span


def setup_observability(*, service_name: str | None = None) -> None:
    """初始化 logging + tracing（幂等，编排入口调用一次即可）。"""
    configure_logging()
    configure_tracing(service_name=service_name)


__all__ = [
    "configure_logging",
    "configure_tracing",
    "get_traces_output_dir",
    "get_logger",
    "get_trace_ids",
    "log_info",
    "record_exception",
    "record_tool_event",
    "setup_observability",
    "shutdown_tracing",
    "span",
]
