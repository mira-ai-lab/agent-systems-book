"""A2A 远程 Agent 端点声明（Supervisor handoff 目标）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class A2AEndpoint:
    """单个远程 A2A 子智能体。

    ``registry_agent`` 非空时，在 ``transport=mixed`` 下用远程替代同名本地 Agent。
    """

    node_name: str
    url: str
    description: str = ""
    registry_agent: Optional[str] = None

    def is_configured(self) -> bool:
        return bool((self.url or "").strip())
