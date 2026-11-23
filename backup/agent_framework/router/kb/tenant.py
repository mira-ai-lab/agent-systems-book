"""知识库多租户 scope：shared + tenant overlay。"""

from __future__ import annotations

from typing import Optional

SHARED_TENANT_ID = "default"


def normalize_kb_tenant_id(user_id: Optional[str]) -> str:
    uid = (user_id or SHARED_TENANT_ID).strip()
    return uid or SHARED_TENANT_ID


def is_shared_kb_tenant(tenant_id: Optional[str]) -> bool:
    return normalize_kb_tenant_id(tenant_id) == SHARED_TENANT_ID
