"""
可观测性包（Observability）：结构化 logging + OpenTelemetry tracing。

模块职责：
    logging_config.py  — 日志格式、trace_id/span_id 自动注入
    setup.py           — TracerProvider 初始化与导出器选择
    trace_provider.py  — @trace_span 装饰器、采样、序列化、业务 event
    spans.py           — span() / record_exception / record_tool_event 兼容层
    file_exporter.py   — span 写入本地 JSONL

Span 命名规范（latc）：
    latc.{OTEL_SERVICE_NAME}.request
    latc.{OTEL_SERVICE_NAME}.orchestration.pre_survey
    latc.{OTEL_SERVICE_NAME}.agent.invoke

环境变量：
  OTEL_SERVICE_NAME          服务名（默认 travel-multi-agent）
  OTEL_TRACES_EXPORTER       console | file | otlp | none（默认 console）
  OTEL_TRACES_DIR            file 模式输出目录（默认 Chapter-8/traces/）
  OTEL_TRACES_FILE_MODE      timestamp | append（默认 timestamp，按启动时间分文件）
  OTEL_TRACES_FILENAME       显式指定文件名（如 debug_run.jsonl）
  OTEL_EXPORTER_OTLP_ENDPOINT OTLP 地址（默认 http://localhost:4317）
  OTEL_TRACES_SAMPLE_ALL     1 时所有 span 采样（开发 / 测试）
  OTEL_TRACE_ATTR_MAX_LEN    attrs 截断长度（默认 500）
  OTEL_TRACE_RESULT_MAX_LEN  result event 截断（默认 2000）
  LOG_LEVEL                  日志级别（默认 INFO）
  LOG_JSON                   true 时输出 JSON 行日志
"""

from agent_framework.tracing.logging_config import (
    configure_logging,
    get_logger,
    get_trace_ids,
    log_info,
)
from agent_framework.tracing.setup import (
    configure_tracing,
    get_traces_output_dir,
    get_traces_output_file,
    shutdown_tracing,
)
from agent_framework.tracing.spans import record_exception, record_tool_event, span
from agent_framework.tracing.trace_provider import (
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
