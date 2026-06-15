"""OpenTelemetry TracerProvider 初始化与导出器装配。

导出模式（OTEL_TRACES_EXPORTER）：
    console  — 打印到终端（默认）
    file     — 写入 Chapter-8/traces/*.jsonl
    otlp     — 发往 OTLP Collector（如 Jaeger / Tempo）
    none     — 不导出 span
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from agent_framework.config import TRACES_DIR
from agent_framework.tracing.file_exporter import FileSpanExporter
from agent_framework.tracing.logging_config import get_logger, log_info
from agent_framework.tracing.trace_provider import build_sampler

logger = get_logger(__name__)

_initialized = False
_file_exporter: FileSpanExporter | None = None


def get_traces_output_dir() -> Path | None:
    """file 模式下 span 写入目录；未启用 file 导出时返回 None。"""
    if _file_exporter is None:
        return None
    return _file_exporter.output_dir


def get_traces_output_file() -> Path | None:
    """file 模式下当前进程正在写入的 jsonl 文件路径。"""
    if _file_exporter is None:
        return None
    return _file_exporter.output_file


def configure_tracing(*, service_name: Optional[str] = None) -> None:
    """配置全局 TracerProvider（幂等，进程内只初始化一次）。"""
    global _initialized, _file_exporter
    if _initialized:
        return

    name = service_name or os.getenv("OTEL_SERVICE_NAME", "travel-multi-agent")
    exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "console").strip().lower()

    provider = TracerProvider(
        resource=Resource.create({"service.name": name}),
        sampler=build_sampler(),  # 默认仅 latc.* 前缀 root span 采样
    )

    if exporter in ("none", "off", "disabled"):
        pass
    elif exporter in ("file", "dir", "local"):
        traces_dir = Path(
            os.getenv("OTEL_TRACES_DIR", str(TRACES_DIR))
        ).expanduser()
        _file_exporter = FileSpanExporter(traces_dir)
        provider.add_span_processor(BatchSpanProcessor(_file_exporter))
        log_info(
            logger,
            "tracing.file_exporter",
            output_dir=str(_file_exporter.output_dir),
            output_file=str(_file_exporter.output_file),
        )
    elif exporter in ("otlp", "otlp/grpc"):
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
    else:
        # 未知配置回退 console，避免静默丢失 trace
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _initialized = True


def shutdown_tracing() -> None:
    """刷新 BatchSpanProcessor 缓冲并关闭 TracerProvider（进程退出前可选调用）。"""
    global _initialized, _file_exporter
    if not _initialized:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    _initialized = False
    _file_exporter = None
