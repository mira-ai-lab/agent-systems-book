"""平台 Prometheus 指标（可选 prometheus-client）。"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from agent_framework.observability.request_context import current_metrics_labels

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    CHAT_REQUESTS = Counter(
        "agent_platform_chat_requests_total",
        "Total /v1/chat requests",
        ["domain", "mode", "transport", "status"],
    )
    JOB_REQUESTS = Counter(
        "agent_platform_job_requests_total",
        "Total /v1/jobs submissions",
        ["domain", "mode", "transport"],
    )
    JOB_OUTCOMES = Counter(
        "agent_platform_job_outcomes_total",
        "Async job execution outcomes",
        ["domain", "mode", "transport", "status"],
    )
    A2A_CALLS = Counter(
        "agent_platform_a2a_calls_total",
        "A2A remote agent calls",
        ["domain", "endpoint", "status"],
    )
    A2A_CALL_DURATION = Histogram(
        "agent_platform_a2a_call_duration_seconds",
        "A2A remote call latency",
        ["domain", "endpoint"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    HANDOFFS = Counter(
        "agent_platform_handoffs_total",
        "Supervisor handoff to sub-agents",
        ["domain", "target", "transport"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain"
    _PROMETHEUS_AVAILABLE = False
    CHAT_REQUESTS = None
    JOB_REQUESTS = None
    JOB_OUTCOMES = None
    A2A_CALLS = None
    A2A_CALL_DURATION = None
    HANDOFFS = None


def metrics_enabled() -> bool:
    return _PROMETHEUS_AVAILABLE


def endpoint_label(url: str) -> str:
    """低基数 endpoint 标签（host:port）。"""
    raw = (url or "").strip()
    if not raw:
        return "unknown"
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    return parsed.netloc or raw.rstrip("/")[:64] or "unknown"


def record_chat(
    domain: str,
    status: str,
    *,
    mode: str = "fixed_graph",
    transport: str = "local",
) -> None:
    if CHAT_REQUESTS is not None:
        CHAT_REQUESTS.labels(
            domain=domain,
            mode=mode,
            transport=transport,
            status=status,
        ).inc()


def record_job(
    domain: str,
    *,
    mode: str = "fixed_graph",
    transport: str = "local",
) -> None:
    if JOB_REQUESTS is not None:
        JOB_REQUESTS.labels(domain=domain, mode=mode, transport=transport).inc()


def record_job_outcome(
    domain: str,
    status: str,
    *,
    mode: str = "fixed_graph",
    transport: str = "local",
) -> None:
    if JOB_OUTCOMES is not None:
        JOB_OUTCOMES.labels(
            domain=domain,
            mode=mode,
            transport=transport,
            status=status,
        ).inc()


def record_a2a_call(
    endpoint: str,
    *,
    status: str,
    duration_sec: float,
    domain: Optional[str] = None,
) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    dom = (domain if domain is not None else current_metrics_labels().get("domain", "")) or "unknown"
    ep = endpoint_label(endpoint)
    if A2A_CALLS is not None:
        A2A_CALLS.labels(domain=dom, endpoint=ep, status=status).inc()
    if A2A_CALL_DURATION is not None:
        A2A_CALL_DURATION.labels(domain=dom, endpoint=ep).observe(max(duration_sec, 0.0))


def record_handoff(
    target: str,
    transport: str,
    *,
    domain: Optional[str] = None,
) -> None:
    if HANDOFFS is None:
        return
    dom = (domain if domain is not None else current_metrics_labels().get("domain", "")) or "unknown"
    HANDOFFS.labels(
        domain=dom,
        target=(target or "unknown").strip() or "unknown",
        transport=(transport or "local").strip() or "local",
    ).inc()


def render_metrics() -> tuple[bytes, str]:
    if not _PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain; charset=utf-8"
    return generate_latest(), CONTENT_TYPE_LATEST
