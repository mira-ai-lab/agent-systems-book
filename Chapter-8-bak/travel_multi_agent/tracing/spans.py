"""Span 工具：节点 / Agent / 工具调用的统一埋点。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from travel_multi_agent.tracing.logging_config import get_logger, log_info

logger = get_logger(__name__)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("travel_multi_agent")


def _set_attrs(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """创建当前 context 下的 OTel span。"""
    with get_tracer().start_as_current_span(name) as current:
        _set_attrs(current, attributes)
        log_info(logger, "span.start", span=name, **attributes)
        try:
            yield current
            current.set_status(Status(StatusCode.OK))
            log_info(logger, "span.ok", span=name)
        except Exception as exc:
            attrs = dict(attributes)
            step = attrs.pop("step", name)
            record_exception(exc, step=step, **attrs)
            raise


def record_exception(
    exc: BaseException,
    *,
    step: Optional[str] = None,
    **attributes: Any,
) -> None:
    """在当前 span 上记录异常并标记 ERROR。"""
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
    """在 agent span 上记录一次工具调用（LangChain tool message）。"""
    current = trace.get_current_span()
    attrs = {
        "tool.name": tool_name,
        "task.id": task_id,
        "agent.name": agent_name,
        "tool.has_error": has_error,
    }
    if output_preview:
        attrs["tool.output_preview"] = output_preview[:500]
    current.add_event("tool.completed", attrs)
    log_info(
        logger,
        "tool.completed",
        tool=tool_name,
        agent=agent_name,
        task_id=task_id,
        has_error=has_error,
    )
