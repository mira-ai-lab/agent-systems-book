"""OpenTelemetry TracerProvider 初始化。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from travel_multi_agent.config import TRACES_DIR
from travel_multi_agent.tracing.file_exporter import FileSpanExporter
from travel_multi_agent.tracing.logging_config import get_logger, log_info
from travel_multi_agent.tracing.trace_provider import build_sampler

logger = get_logger(__name__)

_initialized = False
_file_exporter: FileSpanExporter | None = None


def get_traces_output_dir() -> Path | None:
    """file 模式下 span 写入目录；未启用时返回 None。"""
    if _file_exporter is None:
        return None
    return _file_exporter.output_dir


def get_traces_output_file() -> Path | None:
    """file 模式下当前进程 span 输出文件；未启用时返回 None。"""
    if _file_exporter is None:
        return None
    return _file_exporter.output_file


def configure_tracing(*, service_name: Optional[str] = None) -> None:
    """配置全局 TracerProvider（幂等）。"""
    global _initialized, _file_exporter
    if _initialized:
        return

    name = service_name or os.getenv("OTEL_SERVICE_NAME", "travel-multi-agent")
    exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "console").strip().lower()

    provider = TracerProvider(
        resource=Resource.create({"service.name": name}),
        sampler=build_sampler(),
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
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _initialized = True


def shutdown_tracing() -> None:
    """刷新并关闭 span 导出（进程退出前可选调用）。"""
    global _initialized, _file_exporter
    if not _initialized:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    _initialized = False
    _file_exporter = None
