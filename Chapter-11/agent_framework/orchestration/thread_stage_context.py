"""Thread 级 stage summary 累积（跨请求上下文）。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agent_framework.config import PROJECT_ROOT


DEFAULT_THREAD_STAGE_PATH = PROJECT_ROOT / "data" / "thread_stage_context.json"


def resolve_thread_stage_path() -> Path:
    custom = (os.getenv("THREAD_STAGE_CONTEXT_PATH") or "").strip()
    if custom:
        return Path(custom)
    return DEFAULT_THREAD_STAGE_PATH


def should_persist_thread_stage() -> bool:
    return os.getenv("THREAD_STAGE_CONTEXT_PERSIST", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass(frozen=True)
class ThreadStageRecord:
    domain: str
    thread_id: str
    last_stage_summary: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "domain": self.domain,
            "thread_id": self.thread_id,
            "last_stage_summary": self.last_stage_summary,
            "updated_at": self.updated_at,
        }


class ThreadStageContextStore:
    """进程内 thread → last_stage_summary 存储。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, ThreadStageRecord] = {}

    @staticmethod
    def _key(domain: str, thread_id: str) -> str:
        return f"{domain.strip()}::{thread_id.strip()}"

    def get_last_stage_summary(self, domain: str, thread_id: str) -> str:
        key = self._key(domain, thread_id)
        with self._lock:
            record = self._records.get(key)
            return str(record.last_stage_summary if record else "")

    def set_last_stage_summary(
        self,
        domain: str,
        thread_id: str,
        summary: str,
    ) -> None:
        text = (summary or "").strip()
        if not text:
            return
        key = self._key(domain, thread_id)
        from datetime import datetime, timezone

        record = ThreadStageRecord(
            domain=domain.strip(),
            thread_id=thread_id.strip(),
            last_stage_summary=text,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._records[key] = record

    def clear_thread(self, domain: str, thread_id: str) -> bool:
        key = self._key(domain, thread_id)
        with self._lock:
            if key not in self._records:
                return False
            del self._records[key]
            return True


class PersistedThreadStageContextStore(ThreadStageContextStore):
    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path or resolve_thread_stage_path()
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            for key, item in payload.items():
                if not isinstance(item, dict):
                    continue
                summary = str(item.get("last_stage_summary") or "").strip()
                if not summary:
                    continue
                self._records[str(key)] = ThreadStageRecord(
                    domain=str(item.get("domain") or ""),
                    thread_id=str(item.get("thread_id") or ""),
                    last_stage_summary=summary,
                    updated_at=str(item.get("updated_at") or ""),
                )

    def _save(self) -> None:
        payload: Dict[str, Dict[str, str]] = {}
        with self._lock:
            for key, record in self._records.items():
                payload[key] = record.to_dict()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def set_last_stage_summary(
        self,
        domain: str,
        thread_id: str,
        summary: str,
    ) -> None:
        super().set_last_stage_summary(domain, thread_id, summary)
        self._save()

    def clear_thread(self, domain: str, thread_id: str) -> bool:
        removed = super().clear_thread(domain, thread_id)
        if removed:
            self._save()
        return removed


_store: ThreadStageContextStore | None = None


def get_thread_stage_store() -> ThreadStageContextStore:
    global _store
    if _store is None:
        if should_persist_thread_stage():
            _store = PersistedThreadStageContextStore()
        else:
            _store = ThreadStageContextStore()
    return _store


def reset_thread_stage_store() -> None:
    global _store
    _store = ThreadStageContextStore()
