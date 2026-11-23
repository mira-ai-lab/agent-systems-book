"""API Key 鉴权（未配置 API_KEYS 时放行，便于本地开发）。"""

from __future__ import annotations

import os
from typing import Optional, Set

from fastapi import Header, HTTPException


def _parse_api_keys() -> Set[str]:
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """校验 X-API-Key；环境变量 API_KEYS 为空则跳过鉴权。"""
    allowed = _parse_api_keys()
    if not allowed:
        return
    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
