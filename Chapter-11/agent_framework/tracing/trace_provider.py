"""latc 规范 tracing 核心：@trace_span 装饰器、采样、参数序列化与业务 event。

典型 span 生命周期（@trace_span 装饰的 async 函数）：
    1. attach_parent_context（可选，用于子 Agent 挂到 execute_layer span 下）
    2. start_as_current_span → span.start event + request attributes
    3. 执行业务函数
    4. span.ok + result event，或 span.error + exception event
    5. otel_context.detach 恢复父 context

与 orchestration 配合：
    orchestrator.process_request  → latc.*.request（根 span）
    nodes.make_nodes 各节点     → latc.*.orchestration.*
    _invoke_sub_agent            → latc.*.agent.invoke（parent_arg=layer_span）
"""

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

from agent_framework.tracing.logging_config import get_logger, log_info

from agent_framework.config import PLATFORM_SERVICE_NAME

LATC_PREFIX = "latc."
DEFAULT_SERVICE_NAME = PLATFORM_SERVICE_NAME
# span attribute / event 截断上限，避免超大 state 撑爆导出器
DEFAULT_ATTR_MAX_LEN = int(os.getenv("OTEL_TRACE_ATTR_MAX_LEN", "500"))
DEFAULT_RESULT_MAX_LEN = int(os.getenv("OTEL_TRACE_RESULT_MAX_LEN", "2000"))
# state 参数序列化白名单：只记录调度相关字段，不 dump 整个 execution_plan
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


# ---------------------------------------------------------------------------
# 采样策略
# ---------------------------------------------------------------------------

def _sample_all_enabled() -> bool:
    return os.getenv("OTEL_TRACES_SAMPLE_ALL", "0").strip().lower() in ("1", "true", "yes")


class LatcPrefixSampler(Sampler):
    """采样策略：默认只记录 latc.* 根 span；已采样父 span 的子 span 跟随采样。

    OTEL_TRACES_SAMPLE_ALL=1 时等价于全量采样（测试 / 开发用）。
    """

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
        # 子 span：父已采样则跟随，保证 trace 树完整
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


# ---------------------------------------------------------------------------
# Span 命名（latc.{service}.{suffix}）
# ---------------------------------------------------------------------------

def get_service_name() -> str:
    """从 OTEL_SERVICE_NAME 读取服务名（与 configure_tracing 默认一致）。"""
    name = (os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME).strip()
    return name or DEFAULT_SERVICE_NAME


def get_span_prefix() -> str:
    """latc 规范 span 前缀：latc.{OTEL_SERVICE_NAME}"""
    return f"{LATC_PREFIX}{get_service_name()}"


def span_name(suffix: str) -> str:
    """拼接完整 span 名，suffix 不含前缀（如 orchestration.pre_survey）。"""
    suffix = suffix.strip(".")
    return f"{get_span_prefix()}.{suffix}"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(get_service_name())


# ---------------------------------------------------------------------------
# 参数 / 结果序列化（写入 span attribute 与 event）
# ---------------------------------------------------------------------------

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
    """将 @trace_span(attrs_args=...) 指定的参数转为可写入 span 的安全值。"""
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
    """函数返回值摘要：只挑关键字段写入 result event，避免 dump 整个 state。"""
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


# ---------------------------------------------------------------------------
# 父 span 上下文传播（构建子 Agent 调用链）
# ---------------------------------------------------------------------------

def attach_parent_context(trace_parent: Any) -> Optional[object]:
    """将子调用挂到显式父 span 下；返回 otel detach token。

    支持类型：Span 对象、 (trace_id, span_id) 元组、W3C traceparent dict、
    或带 trace_id/span_id 属性的 span-like 对象。
    """
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
    """在当前 active span 上追加业务 event（如 plan.built、tool.completed）。"""
    current = trace.get_current_span()
    if not current.get_span_context().is_valid:
        return
    attrs = attributes or {}
    safe = {k: serialize_for_trace(v, max_len=DEFAULT_ATTR_MAX_LEN) for k, v in attrs.items()}
    current.add_event(name, safe)


def inject_trace_context(carrier: dict[str, str]) -> dict[str, str]:
    """将当前 active span 的 W3C trace 上下文写入 carrier（如 HTTP headers）。"""
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def extract_trace_context(carrier: Mapping[str, str]) -> Optional[object]:
    """从 carrier（HTTP headers 等）提取 W3C trace 上下文并 attach；返回 detach token。"""
    if not carrier:
        return None
    normalized = {k.lower(): v for k, v in carrier.items()}
    traceparent = normalized.get("traceparent")
    if not traceparent:
        return None
    return attach_parent_context({"traceparent": traceparent})


# ---------------------------------------------------------------------------
# @trace_span 装饰器内部：参数提取、span 生命周期
# ---------------------------------------------------------------------------

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
    """统一 span 收尾：OK → result event；ERROR → exception + error event。"""
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
    """同步函数 span 包装。"""
    if not name.startswith(LATC_PREFIX) and not _sample_all_enabled():
        log_info(logger, "span.name_warning", span=name, hint="latc. prefix recommended")

    parent_token = _resolve_parent_token(fn, args, kwargs, parent_arg)
    request_payload = _build_request_payload(fn, args, kwargs, attrs_args)

    try:
        with get_tracer().start_as_current_span(name) as current:
            _set_attrs(current, {k.replace("_", "."): v for k, v in request_payload.items() if k != "state"})
            # state / task 展开为扁平 attribute（task.id、state.thread_id 等）
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
            # state / task 展开为扁平 attribute（task.id、state.thread_id 等）
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
    """同步 with 块 span 上下文；spans.span() 委托到此函数。"""
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
    """装饰 sync / async / async generator 函数，自动创建 latc span。

    参数：
        name          完整 span 名，建议 span_name("orchestration.xxx")
        attrs_args    从函数参数中提取并写入 span 的参数名列表
        parent_arg    参数名，其值作为显式父 context（如 trace_parent=layer_span）
        record_result 是否在 span 结束时写 result event
    """

    def decorator(fn: F) -> F:
        if inspect.isasyncgenfunction(fn):
            # async generator（如 iter_request_stream）单独包装

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
