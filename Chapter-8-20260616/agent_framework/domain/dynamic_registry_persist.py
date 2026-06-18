"""动态 Agent Registry JSON 持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from agent_framework.config import PROJECT_ROOT
from agent_framework.domain.dynamic_registry import DynamicAgentRecord, DynamicAgentStore

DEFAULT_DYNAMIC_AGENTS_PATH = PROJECT_ROOT / "data" / "dynamic_agents.json"


def resolve_dynamic_agents_path() -> Path:
    custom = (os.getenv("DYNAMIC_AGENTS_PATH") or "").strip()
    if custom:
        return Path(custom)
    return DEFAULT_DYNAMIC_AGENTS_PATH


def should_persist_dynamic_agents() -> bool:
    return os.getenv("DYNAMIC_AGENTS_PERSIST", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class PersistedDynamicAgentStore(DynamicAgentStore):
    """将动态 Agent 记录持久化到 JSON 文件。"""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path or resolve_dynamic_agents_path()
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
            self._records.clear()
            for domain, agents in payload.items():
                if not isinstance(agents, dict):
                    continue
                bucket: Dict[str, DynamicAgentRecord] = {}
                for name, item in agents.items():
                    if not isinstance(item, dict):
                        continue
                    try:
                        bucket[str(name)] = DynamicAgentRecord.from_dict(item)
                    except ValueError:
                        continue
                if bucket:
                    self._records[str(domain)] = bucket

    def _save(self) -> None:
        payload: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for domain, agents in self._records.items():
                payload[domain] = {name: rec.to_dict() for name, rec in agents.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register(self, domain: str, record: DynamicAgentRecord) -> DynamicAgentRecord:
        saved = super().register(domain, record)
        self._save()
        return saved

    def unregister(self, domain: str, name: str) -> bool:
        removed = super().unregister(domain, name)
        if removed:
            self._save()
        return removed

    def clear_domain(self, domain: str) -> None:
        super().clear_domain(domain)
        self._save()
