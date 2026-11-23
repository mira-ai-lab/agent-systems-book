"""Load / save optimized travel planner prompts for runtime and offline tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent_framework.config import PROJECT_ROOT

OPTIMIZABLE_TRAVEL_PROMPT_FIELDS = ("decomposition_prompt", "agent_routing")
DEFAULT_OPTIMIZED_DIR = PROJECT_ROOT / "data" / "benchmark" / "travel_planner" / "optimized"
LEGACY_OPTIMIZED_DIR = PROJECT_ROOT / "data" / "benchmark" / "travel_decomposition" / "optimized"


def optimized_prompts_enabled() -> bool:
    return os.getenv("TRAVEL_OPTIMIZED_PROMPTS", "1").strip().lower() not in ("0", "false", "no")


def optimized_prompts_path(locale: str = "zh") -> Path:
    override = os.getenv("TRAVEL_OPTIMIZED_PROMPTS_FILE", "").strip()
    if override:
        return Path(override)
    return DEFAULT_OPTIMIZED_DIR / f"{locale}.json"


def _resolve_existing_path(locale: str) -> Optional[Path]:
    primary = optimized_prompts_path(locale)
    if primary.is_file():
        return primary
    legacy = LEGACY_OPTIMIZED_DIR / f"{locale}.json"
    if legacy.is_file():
        return legacy
    return None


def load_optimized_prompt_payload(locale: str = "zh") -> Dict[str, Any]:
    if not optimized_prompts_enabled():
        return {}
    path = _resolve_existing_path(locale)
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_optimized_prompts(locale: str = "zh") -> Dict[str, str]:
    payload = load_optimized_prompt_payload(locale)
    return {
        key: str(payload.get(key) or "").strip()
        for key in OPTIMIZABLE_TRAVEL_PROMPT_FIELDS
        if str(payload.get(key) or "").strip()
    }


def apply_prompt_overrides(prompts, *, locale: str = "zh"):
    from dataclasses import fields, replace

    overrides = load_optimized_prompts(locale)
    if not overrides:
        return prompts
    valid = {name for name in overrides if name in {field.name for field in fields(prompts)}}
    return replace(prompts, **{key: overrides[key] for key in valid})


def save_optimized_prompts(
    path: Path,
    *,
    updates: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    existing: Dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded

    for key, value in updates.items():
        if key in OPTIMIZABLE_TRAVEL_PROMPT_FIELDS and str(value).strip():
            existing[key] = str(value)

    if metadata:
        meta = existing.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta.update(metadata)
        existing["metadata"] = meta

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
