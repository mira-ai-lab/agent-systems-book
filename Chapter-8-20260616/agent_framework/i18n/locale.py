"""Locale 规范化。"""

from __future__ import annotations

from typing import Optional

SUPPORTED_LOCALES = ("zh", "en")
DEFAULT_LOCALE = "zh"


def normalize_locale(locale: Optional[str]) -> str:
    value = (locale or DEFAULT_LOCALE).strip().lower() or DEFAULT_LOCALE
    if value not in SUPPORTED_LOCALES:
        raise ValueError(
            f"不支持的 locale='{value}'，可选: {', '.join(SUPPORTED_LOCALES)}"
        )
    return value
