"""平台可观测性（Prometheus metrics 上下文）。"""

from agent_framework.observability.metrics import (
    endpoint_label,
    metrics_enabled,
    record_a2a_call,
    record_chat,
    record_handoff,
    record_job,
    record_job_outcome,
    render_metrics,
)
from agent_framework.observability.request_context import current_metrics_labels, request_metrics_context

__all__ = [
    "current_metrics_labels",
    "endpoint_label",
    "metrics_enabled",
    "record_a2a_call",
    "record_chat",
    "record_handoff",
    "record_job",
    "record_job_outcome",
    "render_metrics",
    "request_metrics_context",
]
