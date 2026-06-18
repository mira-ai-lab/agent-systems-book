"""平台级默认话术（DomainPrompts 空字段 fallback）。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


@lru_cache(maxsize=8)
def load_platform_locale(locale: str) -> dict[str, Any]:
    loc = (locale or "zh").strip() or "zh"
    path = _LOCALES_DIR / f"{loc}.json"
    if not path.is_file():
        if loc != "zh":
            return load_platform_locale("zh")
        raise FileNotFoundError(f"平台 locale 不存在: {locale} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def get_platform_domain_prompts(locale: str = "zh") -> dict[str, str]:
    data = load_platform_locale(locale)
    block = data.get("domain_prompts", {})
    return {str(k): str(v) for k, v in block.items()}
