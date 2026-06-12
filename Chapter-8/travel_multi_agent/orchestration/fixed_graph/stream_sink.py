"""流式输出桥：将 LLM token 与阶段进度转发到终端或前端。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class StreamSink:
    """单次请求内启用；由 LangGraphOrchestrator 在流式调用前配置回调。"""

    enabled: bool = False
    on_token: Optional[Callable[[str], None]] = field(default=None, repr=False)
    on_progress: Optional[Callable[[str], None]] = field(default=None, repr=False)

    def emit_token(self, text: str) -> None:
        if self.enabled and self.on_token and text:
            self.on_token(text)

    def emit_progress(self, message: str) -> None:
        if self.enabled and self.on_progress and message:
            self.on_progress(message)

    def reset(self) -> None:
        self.enabled = False
        self.on_token = None
        self.on_progress = None
