"""流式输出桥：将 LLM token 与阶段进度转发到终端或前端。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class StreamSink:
    """流式输出桥：将 LLM token 与阶段进度转发到终端或自定义 UI。

    由 LangGraphOrchestrator 在流式调用前通过 on_token / on_progress 注入回调；
    节点内通过 ctx.stream_sink.emit_* 推送，不直接写 stdout。
    """

    enabled: bool = False
    on_token: Optional[Callable[[str], None]] = field(default=None, repr=False)
    on_progress: Optional[Callable[[str], None]] = field(default=None, repr=False)
    on_subtask_completed: Optional[Callable[[Dict[str, Any]], None]] = field(
        default=None,
        repr=False,
    )
    on_subtask_token: Optional[Callable[[str, str, str], None]] = field(
        default=None,
        repr=False,
    )

    def emit_token(self, text: str) -> None:
        """转发 LLM 生成的单个 token（聚合阶段逐字输出）。"""
        if self.enabled and self.on_token and text:
            self.on_token(text)

    def emit_progress(self, message: str) -> None:
        """转发阶段进度（如「预调查完成」「执行第 2 层」）。"""
        if self.enabled and self.on_progress and message:
            self.on_progress(message)

    def emit_subtask_completed(self, result: Dict[str, Any]) -> None:
        """子 Agent 完成时推送结构化结果摘要。"""
        if self.enabled and self.on_subtask_completed and result:
            self.on_subtask_completed(result)

    def emit_subtask_token(self, task_id: str, agent: str, token: str) -> None:
        """子 Agent LLM 逐 token 推送。"""
        if self.enabled and self.on_subtask_token and token:
            self.on_subtask_token(task_id, agent, token)

    def reset(self) -> None:
        """请求结束后清理回调，避免泄漏到下一次调用。"""
        self.enabled = False
        self.on_token = None
        self.on_progress = None
        self.on_subtask_completed = None
        self.on_subtask_token = None
