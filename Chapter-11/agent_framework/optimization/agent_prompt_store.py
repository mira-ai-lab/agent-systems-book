"""子 Agent optimized system_prompt 的加载与持久化。

与 ``prompt_store.py``（Planner prompts）并列，运行时由 ``travel_agent_prompt()`` 读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent_framework.config import PROJECT_ROOT

DEFAULT_OPTIMIZED_AGENT_DIR = PROJECT_ROOT / "data" / "benchmark" / "travel_agents" / "optimized"


def optimized_agent_prompts_enabled() -> bool:
    """是否启用 optimized agent prompts（可用环境变量关闭）。"""
    return os.getenv("TRAVEL_OPTIMIZED_AGENT_PROMPTS", "1").strip().lower() not in ("0", "false", "no")


def optimized_agent_prompts_path(locale: str = "zh") -> Path:
    override = os.getenv("TRAVEL_OPTIMIZED_AGENT_PROMPTS_FILE", "").strip()
    if override:
        return Path(override)
    return DEFAULT_OPTIMIZED_AGENT_DIR / f"{locale}.json"


def load_optimized_agent_prompt_payload(locale: str = "zh") -> Dict[str, Any]:
    if not optimized_agent_prompts_enabled():
        return {}
    path = optimized_agent_prompts_path(locale)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_optimized_agent_prompt_template(agent_name: str, *, locale: str = "zh") -> str:
    """读取某 Agent 的 optimized system_prompt 模板（未 format）。"""
    payload = load_optimized_agent_prompt_payload(locale)
    agents = payload.get("agents") or {}
    if not isinstance(agents, dict):
        return ""
    entry = agents.get(agent_name) or {}
    if isinstance(entry, dict):
        return str(entry.get("system_prompt") or "").strip()
    return str(entry or "").strip()


def save_optimized_agent_prompts(
    path: Path,
    *,
    agent_name: str,
    system_prompt_template: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存单个 Agent 的 optimized prompt，合并已有文件内容。"""
    existing: Dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded

    agents = existing.get("agents")
    if not isinstance(agents, dict):
        agents = {}
    agents[agent_name] = {"system_prompt": system_prompt_template}
    existing["agents"] = agents

    if metadata:
        meta = existing.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta.update(metadata)
        existing["metadata"] = meta

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
