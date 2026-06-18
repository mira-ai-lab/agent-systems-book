"""LangGraph checkpoint 后端解析。"""

from __future__ import annotations

import os
from typing import Any, Literal

from agent_framework.config import PROJECT_ROOT

CheckpointBackend = Literal["memory", "sqlite"]


def resolve_checkpoint_backend(explicit: str | None = None) -> CheckpointBackend:
    raw = (explicit or os.getenv("CHECKPOINT_BACKEND", "memory")).lower().strip()
    if raw in ("sqlite", "db", "persist"):
        return "sqlite"
    return "memory"


def resolve_checkpointer(explicit_backend: str | None = None) -> Any:
    """返回 LangGraph checkpointer 实例。"""
    backend = resolve_checkpoint_backend(explicit_backend)
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = os.getenv(
            "CHECKPOINT_SQLITE_PATH",
            str(PROJECT_ROOT / "checkpoints.db"),
        )
        return SqliteSaver.from_conn_string(db_path)

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
