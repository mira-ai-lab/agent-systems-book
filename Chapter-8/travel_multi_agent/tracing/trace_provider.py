"""latc 规范 tracing：@trace_span、采样、参数序列化与业务 event。"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Optional, Sequence, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
    StaticSampler,
)
from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from travel_multi_agent.tracing.logging_config import get_logger, log_info

LATC_PREFIX = "latc."
DEFAULT_ATTR_MAX_LEN = int(os.getenv("OTEL_TRACE_ATTR_MAX_LEN", "500"))
DEFAULT_RESULT_MAX_LEN = int(os.getenv("OTEL_TRACE_RESULT_MAX_LEN", "2000"))
STATE_TRACE_KEYS = (
    "thread_id",
    "user_query",
    "enable_memory",
    "enable_stream",
    "current_layer_index",
)
SENSITIVE_KEY_FRAGMENTS = ("api_key", "token", "password", "secret", "authorization")

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def _sample_all_enabled() -> bool:
    return os.getenv("OTEL_TRACES_SAMPLE_ALL", "0").strip().lower() in ("1", "true", "yes")


class LatcPrefixSampler(Sampler):
    """仅 latc.* 前缀 span 采样；OTEL_TRACES_SAMPLE_ALL=1 时全部采样。"""

    def __init__(self, *, sample_all: bool = False) -> None:
        self._sample_all = sample_all

    def should_sample(
        self,
        parent_context: Optional[otel_context.Context],
        trace_id: int,
        name: str,
        kind: Optional[SpanKind] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        links: Optional[Sequence[Link]] = None,
        trace_state: Optional[trace.TraceState] = None,
    ) -> SamplingResult:
        del trace_id, kind, attributes, links, trace_state
        if self._sample_all or name.startswith(LATC_PREFIX):
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        if parent_context is not None:
            parent_span = trace.get_current_span(parent_context)
            parent_ctx = parent_span.get_span_context()
            if parent_ctx.is_valid and parent_ctx.trace_flags.sampled:
                return SamplingResult(Decision.RECORD_AND_SAMPLE)
        return SamplingResult(Decision.DROP)

    def get_description(self) -> str:
        return "LatcPrefixSampler"


def build_sampler(*, sample_latc_only: bool = True) -> Sampler:
    """构建 TracerProvider 使用的 Sampler。"""
    sample_all = _sample_all_enabled()
    if not sample_latc_only or sample_all:
        return ParentBased(root=StaticSampler(Decision.RECORD_AND_SAMPLE))
    latc = LatcPrefixSampler(sample_all=False)
    return ParentBased(
        root=latc,
        local_parent_sampled=latc,
        remote_parent_sampled=latc,
    )


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("travel_multi_agent")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _filter_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _filter_sensitive(v)
            for k, v in obj.items()
            if not any(s in k.lower() for s in SENSITIVE_KEY_FRAGMENTS)
        }
    if isinstance(obj, list):
        return [_filter_sensitive(v) for v in obj]
    return obj


def _serialize_state(state: Any, *, max_len: int) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"state": _truncate(str(state), max_len)}
    picked = {k: state[k] for k in STATE_TRACE_KEYS if k in state}
    if "user_query" in picked and isinstance(picked["user_query"], str):
        picked["user_query"] = _truncate(picked["user_query"], 200)
    return _filter_sensitive(picked)


def _serialize_task(task: Any, *, max_len: int) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {"task": _truncate(str(task), max_len)}
    desc = task.get("description") or ""
    return _filter_sensitive({
        "task.id": task.get("task_id"),
        "task.agent": task.get("agent"),
        "task.description": _truncate(str(desc), 120),
        "task.depends_on": task.get("depends_on", []),
    })


def serialize_for_trace(value: Any, *, param_name: str = "", max_len: int = DEFAULT_ATTR_MAX_LEN) -> Any:
    """将函数参数序列化为可写入 span attribute / event 的值。"""
    if param_name == "state":
        return _serialize_state(value, max_len=max_len)
    if param_name == "task":
        return _serialize_task(value, max_len=max_len)
    if value is None:
        return None
    if isinstance(value, str):
        return _truncate(value, max_len)
    if isinstance(value, (bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _truncate(json.dumps(_filter_sensitive(asdict(value)), ensure_ascii=False), max_len)
    if hasattr(value, "model_dump"):
        return _truncate(
            json.dumps(_filter_sensitive(value.model_dump()), ensure_ascii=False),
            max_len,
        )
    if isinstance(value, dict):
        return _truncate(json.dumps(_filter_sensitive(value), ensure_ascii=False), max_len)
    if isinstance(value, list):
        return _truncate(json.dumps(_filter_sensitive(value), ensure_ascii=False), max_len)
    return _truncate(str(value), max_len)


def _set_attrs(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _serialize_result(value: Any, *, max_len: int) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        for key in ("status", "task_id", "agent", "final_response", "thread_id"):
            if key in value:
                preview[key] = value[key]
        if "final_response" in preview and isinstance(preview["final_response"], str):
            preview["final_response.length"] = len(preview["final_response"])
            preview["final_response"] = _truncate(preview["final_response"], 200)
        if not preview:
            preview = {"keys": list(value.keys())[:20]}
        text = json.dumps(_filter_sensitive(preview), ensure_ascii=False)
    else:
        text = str(value)
    return _truncate(text, max_len)


def get_current_span_context() -> tuple[str | None, str | None]:
    """返回 (trace_id, span_id) 十六进制字符串。"""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def attach_parent_context(trace_parent: Any) -> Optional[object]:
    """绑定显式父 span context；返回 detach token。"""
    if trace_parent is None:
        return None
    if isinstance(trace_parent, Span):
        ctx = trace.set_span_in_context(trace_parent)
        return otel_context.attach(ctx)
    if isinstance(trace_parent, tuple) and len(trace_parent) == 2:
        trace_id_hex, span_id_hex = trace_parent
        return _attach_from_ids(trace_id_hex, span_id_hex)
    if isinstance(trace_parent, dict) and "traceparent" in trace_parent:
        propagator = TraceContextTextMapPropagator()
        ctx = propagator.extract(trace_parent)
        return otel_context.attach(ctx)
    if hasattr(trace_parent, "trace_id") and hasattr(trace_parent, "span_id"):
        return _attach_from_ids(
            format(trace_parent.trace_id, "032x"),
            format(trace_parent.span_id, "016x"),
        )
    return None


def _attach_from_ids(trace_id_hex: str, span_id_hex: str) -> Optional[object]:
    try:
        trace_id = int(trace_id_hex, 16)
        span_id = int(span_id_hex, 16)
    except (TypeError, ValueError):
        return None
    span_context = trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    ctx = trace.set_span_in_context(trace.NonRecordingSpan(span_context))
    return otel_context.attach(ctx)


def current_trace_add_event(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """在当前 active span 上追加业务 event。"""
    current = trace.get_current_span()
    if not current.get_span_context().is_valid:
        return
    attrs = attributes or {}
    safe = {k: serialize_for_trace(v, max_len=DEFAULT_ATTR_MAX_LEN) for k, v in attrs.items()}
    current.add_event(name, safe)


def _extract_bound_args(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    sig = inspect.signature(fn)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _build_request_payload(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    attrs_args: list[str] | None,
) -> dict[str, Any]:
    if not attrs_args:
        return {}
    bound = _extract_bound_args(fn, args, kwargs)
    payload: dict[str, Any] = {}
    for name in attrs_args:
        if name not in bound.arguments:
            continue
        payload[name] = serialize_for_trace(bound.arguments[name], param_name=name)
    return payload


def _resolve_parent_token(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent_arg: str | None,
) -> Optional[object]:
    if not parent_arg:
        return None
    bound = _extract_bound_args(fn, args, kwargs)
    parent = bound.arguments.get(parent_arg)
    return attach_parent_context(parent)


def _finish_span(
    current: Span,
    name: str,
    *,
    record_result: bool = True,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
    result: Any = None,
    exc: BaseException | None = None,
) -> None:
    if exc is not None:
        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, str(exc)))
        current_trace_add_event(
            "error",
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": _truncate(str(exc), DEFAULT_ATTR_MAX_LEN),
            },
        )
        log_info(
            logger,
            "span.error",
            span=name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return
    current.set_status(Status(StatusCode.OK))
    if record_result:
        current_trace_add_event(
            "result",
            {"status": "ok", "preview": _serialize_result(result, max_len=result_max_len)},
        )
    log_info(logger, "span.ok", span=name)


def _run_with_span(
    name: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    attrs_args: list[str] | None,
    parent_arg: str | None,
    record_result: bool,
    result_max_len: int,
) -> Any:
    if not name.startswith(LATC_PREFIX) and not _sample_all_enabled():
        log_info(logger, "span.name_warning", span=name, hint="latc. prefix recommended")

    parent_token = _resolve_parent_token(fn, args, kwargs, parent_arg)
    request_payload = _build_request_payload(fn, args, kwargs, attrs_args)

    try:
        with get_tracer().start_as_current_span(name) as current:
            _set_attrs(current, {k.replace("_", "."): v for k, v in request_payload.items() if k != "state"})
            if "state" in request_payload and isinstance(request_payload["state"], dict):
                for k, v in request_payload["state"].items():
                    attr_key = k if "." in k else f"state.{k}"
                    _set_attrs(current, {attr_key: v})
            if "task" in request_payload and isinstance(request_payload["task"], dict):
                _set_attrs(current, request_payload["task"])
            current_trace_add_event("request", request_payload)
            log_info(logger, "span.start", span=name, **{k: str(v)[:120] for k, v in request_payload.items()})
            try:
                result = fn(*args, **kwargs)
                _finish_span(
                    current,
                    name,
                    record_result=record_result,
                    result_max_len=result_max_len,
                    result=result,
                )
                return result
            except BaseException as exc:
                _finish_span(current, name, record_result=False, exc=exc)
                raise
    finally:
        if parent_token is not None:
            otel_context.detach(parent_token)


async def _run_with_span_async(
    name: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    attrs_args: list[str] | None,
    parent_arg: str | None,
    record_result: bool,
    result_max_len: int,
) -> Any:
    if not name.startswith(LATC_PREFIX) and not _sample_all_enabled():
        log_info(logger, "span.name_warning", span=name, hint="latc. prefix recommended")

    parent_token = _resolve_parent_token(fn, args, kwargs, parent_arg)
    request_payload = _build_request_payload(fn, args, kwargs, attrs_args)

    try:
        with get_tracer().start_as_current_span(name) as current:
            _set_attrs(current, {k.replace("_", "."): v for k, v in request_payload.items() if k != "state"})
            if "state" in request_payload and isinstance(request_payload["state"], dict):
                for k, v in request_payload["state"].items():
                    attr_key = k if "." in k else f"state.{k}"
                    _set_attrs(current, {attr_key: v})
            if "task" in request_payload and isinstance(request_payload["task"], dict):
                _set_attrs(current, request_payload["task"])
            current_trace_add_event("request", request_payload)
            log_info(logger, "span.start", span=name, **{k: str(v)[:120] for k, v in request_payload.items()})
            try:
                result = await fn(*args, **kwargs)
                _finish_span(
                    current,
                    name,
                    record_result=record_result,
                    result_max_len=result_max_len,
                    result=result,
                )
                return result
            except BaseException as exc:
                _finish_span(current, name, record_result=False, exc=exc)
                raise
    finally:
        if parent_token is not None:
            otel_context.detach(parent_token)


@contextmanager
def start_span_context(name: str, **attributes: Any) -> Iterator[Span]:
    """同步 span 上下文（span() 兼容层委托）。"""
    with get_tracer().start_as_current_span(name) as current:
        _set_attrs(current, attributes)
        log_info(logger, "span.start", span=name, **attributes)
        try:
            yield current
            current.set_status(Status(StatusCode.OK))
            log_info(logger, "span.ok", span=name)
        except BaseException as exc:
            attrs = dict(attributes)
            step = attrs.pop("step", name)
            if step:
                current.set_attribute("error.step", step)
            _set_attrs(current, {f"error.{k}": v for k, v in attrs.items()})
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)))
            log_info(
                logger,
                "span.error",
                span=step or name,
                error_type=type(exc).__name__,
                error_message=str(exc),
                **attrs,
            )
            raise


def trace_span(
    name: str,
    *,
    attrs_args: list[str] | None = None,
    parent_arg: str | None = None,
    record_result: bool = True,
    result_max_len: int = DEFAULT_RESULT_MAX_LEN,
) -> Callable[[F], F]:
    """装饰 async 函数 / async generator，自动创建 latc span。"""

    def decorator(fn: F) -> F:
        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                if not name.startswith(LATC_PREFIX) and not _sample_all_enabled():
                    log_info(logger, "span.name_warning", span=name, hint="latc. prefix recommended")
                parent_token = _resolve_parent_token(fn, args, kwargs, parent_arg)
                request_payload = _build_request_payload(fn, args, kwargs, attrs_args)
                try:
                    with get_tracer().start_as_current_span(name) as current:
                        _set_attrs(current, {k: v for k, v in request_payload.items()})
                        current_trace_add_event("request", request_payload)
                        log_info(logger, "span.start", span=name)
                        try:
                            async for item in fn(*args, **kwargs):
                                yield item
                            _finish_span(current, name, record_result=record_result, result_max_len=result_max_len)
                        except BaseException as exc:
                            _finish_span(current, name, record_result=False, exc=exc)
                            raise
                finally:
                    if parent_token is not None:
                        otel_context.detach(parent_token)

            return async_gen_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_with_span_async(
                    name,
                    fn,
                    args,
                    kwargs,
                    attrs_args=attrs_args,
                    parent_arg=parent_arg,
                    record_result=record_result,
                    result_max_len=result_max_len,
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_with_span(
                name,
                fn,
                args,
                kwargs,
                attrs_args=attrs_args,
                parent_arg=parent_arg,
                record_result=record_result,
                result_max_len=result_max_len,
            )

        return sync_wrapper  # type: ignore[return-value]

    return decorator
