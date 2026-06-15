"""流式输出桥：将 LLM token 与阶段进度转发到终端或前端。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class StreamSink:
    """流式输出桥：将 LLM token 与阶段进度转发到终端或自定义 UI。

    由 LangGraphOrchestrator 在流式调用前通过 on_token / on_progress 注入回调；
    节点内通过 ctx.stream_sink.emit_* 推送，不直接写 stdout。
    """

    enabled: bool = False
    on_token: Optional[Callable[[str], None]] = field(default=None, repr=False)
    on_progress: Optional[Callable[[str], None]] = field(default=None, repr=False)

    def emit_token(self, text: str) -> None:
        """转发 LLM 生成的单个 token（聚合阶段逐字输出）。"""
        if self.enabled and self.on_token and text:
            self.on_token(text)

    def emit_progress(self, message: str) -> None:
        """转发阶段进度（如「预调查完成」「执行第 2 层」）。"""
        if self.enabled and self.on_progress and message:
            self.on_progress(message)

    def reset(self) -> None:
        """请求结束后清理回调，避免泄漏到下一次调用。"""
        self.enabled = False
        self.on_token = None
        self.on_progress = None
