"""Prometheus 指标（API 层 re-export，实现见 agent_framework.observability.metrics）。"""

from __future__ import annotations

from agent_framework.observability.metrics import (
    CONTENT_TYPE_LATEST,
    metrics_enabled,
    record_chat,
    record_job,
    record_job_outcome,
    render_metrics,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "metrics_enabled",
    "record_chat",
    "record_job",
    "record_job_outcome",
    "render_metrics",
]
