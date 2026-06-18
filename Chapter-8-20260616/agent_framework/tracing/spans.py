"""Span 工具函数：节点 / Agent / 工具调用的手工埋点（兼容层）。

推荐优先使用 @trace_span 装饰器（trace_provider.py）；
本模块的 span() / record_* 适用于 with 块或异常捕获等场景。
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from agent_framework.tracing.logging_config import get_logger, log_info
from agent_framework.tracing.trace_provider import start_span_context

logger = get_logger(__name__)


def _set_attrs(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def span(name: str, **attributes: Any) -> Iterator[Span]:
    """同步 with 块创建 span（委托 trace_provider.start_span_context）。"""
    return start_span_context(name, **attributes)


def record_exception(
    exc: BaseException,
    *,
    step: Optional[str] = None,
    **attributes: Any,
) -> None:
    """在当前 active span 上记录异常，标记 ERROR 并写结构化日志。"""
    current = trace.get_current_span()
    if step:
        current.set_attribute("error.step", step)
    _set_attrs(current, {f"error.{k}": v for k, v in attributes.items()})
    current.record_exception(exc)
    current.set_status(Status(StatusCode.ERROR, str(exc)))
    log_info(
        logger,
        "span.error",
        span=step or "unknown",
        error_type=type(exc).__name__,
        error_message=str(exc),
        **attributes,
    )


def record_tool_event(
    tool_name: str,
    *,
    task_id: str,
    agent_name: str,
    has_error: bool = False,
    output_preview: Optional[str] = None,
) -> None:
    """在 agent.invoke span 上追加 tool.completed 业务 event（LangChain tool message）。"""
    from agent_framework.tracing.trace_provider import current_trace_add_event

    attrs: dict[str, Any] = {
        "tool.name": tool_name,
        "task.id": task_id,
        "agent.name": agent_name,
        "tool.has_error": has_error,
    }
    if output_preview:
        attrs["tool.output_preview"] = output_preview[:500]
    current_trace_add_event("tool.completed", attrs)
    log_info(
        logger,
        "tool.completed",
        tool=tool_name,
        agent=agent_name,
        task_id=task_id,
        has_error=has_error,
    )
