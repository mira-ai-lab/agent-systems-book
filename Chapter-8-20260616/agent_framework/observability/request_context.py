"""请求级 metrics 上下文（domain / mode / transport）。"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH, MODE_SUPERVISOR

_ctx: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "request_metrics_ctx",
    default={"domain": "", "mode": MODE_FIXED_GRAPH, "transport": "local"},
)


def _normalize_mode(mode: str | None) -> str:
    value = (mode or MODE_FIXED_GRAPH).strip() or MODE_FIXED_GRAPH
    return value if value in (MODE_FIXED_GRAPH, MODE_SUPERVISOR) else MODE_FIXED_GRAPH


def _normalize_transport(mode: str, transport: str | None) -> str:
    if mode != MODE_SUPERVISOR:
        return "local"
    value = (transport or "local").strip() or "local"
    return value if value in ("local", "a2a", "mixed") else "local"


def current_metrics_labels() -> dict[str, str]:
    return dict(_ctx.get())


@contextmanager
def request_metrics_context(
    *,
    domain: str = "",
    mode: str = MODE_FIXED_GRAPH,
    transport: str = "local",
) -> Iterator[None]:
    normalized_mode = _normalize_mode(mode)
    token = _ctx.set(
        {
            "domain": (domain or "").strip(),
            "mode": normalized_mode,
            "transport": _normalize_transport(normalized_mode, transport),
        }
    )
    try:
        yield
    finally:
        _ctx.reset(token)
