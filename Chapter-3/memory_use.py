"""Memory use: build short-term context and long-term-memory prompts.

This module converts retrieved profile facts and preference memories into
model-ready context and maintains short-term dialogue buffers isolated by
thread.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

from typing import Any


MEMORY_PROMPT_TEMPLATE = """你是一个具备长期记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

【最近对话】（按时间顺序）
{recent_dialogue}

【相关长期记忆】（用户画像事实与偏好记忆）
{memory_context}

【当前问题】
{query}
"""


class ThreadShortTermMemory:
    """Short-term dialogue buffer isolated by thread."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._threads: dict[str, list[dict[str, str]]] = {}

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to a thread and retain the latest ``max_turns``."""
        self._threads.setdefault(thread_id, []).append(
            {"role": role, "content": content}
        )
        if len(self._threads[thread_id]) > self.max_turns:
            self._threads[thread_id] = self._threads[thread_id][-self.max_turns :]

    def format_recent(self, thread_id: str, last_n: int = 6) -> str:
        """Format the most recent messages from a specific thread."""
        turns = self._threads.get(thread_id, [])[-last_n:]
        if not turns:
            return "（无历史对话）"
        return "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in turns
        )

class MemoryContextBuilder:
    """Combine short-term dialogue and long-term memory into a final prompt."""

    def __init__(
        self,
        short_term: ThreadShortTermMemory | None = None,
        *,
        short_term_max_turns: int = 20,
    ) -> None:
        self.short_term = short_term or ThreadShortTermMemory(short_term_max_turns)

    def record_turn(
        self,
        thread_id: str,
        user_content: str,
        assistant_content: str | None = None,
    ) -> None:
        """Record one user message and an optional assistant response."""
        self.short_term.append(thread_id, "user", user_content)
        if assistant_content:
            self.short_term.append(thread_id, "assistant", assistant_content)

    def build_prompt(
        self,
        thread_id: str,
        query: str,
        *,
        hits: list[dict[str, Any]],
        recent_n: int = 6,
    ) -> str:
        """Build a prompt containing short-term dialogue and profile memory."""
        if hits:
            memory_context = "\n".join(
                f"- [{hit.get('memory_type', 'general')}] "
                f"{hit.get('summary') or hit.get('content')} "
                f"(final={hit.get('final_score', 0.0):.4f}, "
                f"vector_sim={hit.get('vector_sim', 0.0):.4f}, "
                f"rrf={hit.get('rrf_score', 0.0):.4f})"
                for hit in hits
            )
        else:
            memory_context = "（未检索到相关长期记忆）"
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=self.short_term.format_recent(thread_id, recent_n),
            memory_context=memory_context,
            query=query,
        )
