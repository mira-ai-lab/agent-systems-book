"""Memory merging: consolidate semantically or textually similar memories.

The merger creates one consolidated memory while preserving each source
memory's unique information. This controls memory size and reduces duplicate
preferences and facts.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable


ChatFn = Callable[[Any, str], dict[str, Any]]
AsyncChatFn = Callable[[Any, str], Awaitable[dict[str, Any]]]


def _default_chat_fn(llm: Any, prompt: str) -> dict[str, Any]:
    from llm import invoke_json_sync

    return invoke_json_sync(llm, prompt, require_object=True)


async def _default_async_chat_fn(llm: Any, prompt: str) -> dict[str, Any]:
    from llm import invoke_json

    return await invoke_json(llm, prompt, require_object=True)


class MemoryMerger:
    """Merge similar memories with an instruction-following LLM."""

    def __init__(
        self,
        llm: Any,
        *,
        chat_fn: ChatFn | None = None,
        async_chat_fn: AsyncChatFn | None = None,
    ) -> None:
        self.llm = llm
        self.chat_fn = chat_fn
        self.async_chat_fn = async_chat_fn

    @staticmethod
    def _format_memories(memories: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"记忆{index + 1}:\n{memory['content']}"
            for index, memory in enumerate(memories)
        )

    @classmethod
    def _build_prompt(cls, memories: list[dict[str, Any]]) -> str:
        memories_text = cls._format_memories(memories)
        return f"""请将以下多个相似记忆合并为一个综合记忆，保留所有独特的重要信息：
        {memories_text}

        只输出一个 JSON 对象：
        {{
            "summary": "合并后的一句话总结",
            "key_points": ["合并后的要点1", "合并后的要点2"],
            "importance": 0.5
        }}"""

    @staticmethod
    def _fallback(memories: list[dict[str, Any]]) -> dict[str, Any]:
        base = memories[0]
        return {
            **base,
            "id": str(uuid.uuid4()),
            "metadata": {
                "merged_from": [memory["id"] for memory in memories],
                **(base.get("metadata") or {}),
            },
            "is_merged": True,
        }

    @staticmethod
    def _build_result(
        merged: dict[str, Any],
        memories: list[dict[str, Any]],
        memories_text: str,
    ) -> dict[str, Any]:
        key_points = list(merged.get("key_points") or [])
        return {
            "id": str(uuid.uuid4()),
            "content": "\n".join(key_points) if key_points else memories_text,
            "summary": str(merged.get("summary", memories[0].get("summary", ""))),
            "key_points": key_points,
            "importance": float(merged.get("importance", 0.5)),
            "timestamp": datetime.now().isoformat(),
            "metadata": {"merged_from": [memory["id"] for memory in memories]},
            "is_merged": True,
        }

    def merge(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge multiple similar memories into one consolidated memory.

        Args:
            memories: Memories to merge. Each item must contain at least the
                ``id``, ``content``, and ``summary`` fields.

        Returns:
            The merged structured memory. If the model is unavailable, a
            fallback copy of the first memory is returned.
        """
        if not memories:
            raise ValueError("At least one memory is required for merging")

        prompt = self._build_prompt(memories)

        try:
            merged = (self.chat_fn or _default_chat_fn)(self.llm, prompt)
        except Exception:
            return self._fallback(memories)
        return self._build_result(
            merged,
            memories,
            self._format_memories(memories),
        )

    async def merge_async(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        """Asynchronously merge similar memories into one consolidated memory."""
        if not memories:
            raise ValueError("At least one memory is required for merging")

        prompt = self._build_prompt(memories)
        try:
            if self.async_chat_fn:
                merged = await self.async_chat_fn(self.llm, prompt)
            else:
                chat_fn = self.chat_fn or _default_chat_fn
                merged = await asyncio.to_thread(chat_fn, self.llm, prompt)
        except Exception:
            return self._fallback(memories)
        return self._build_result(
            merged,
            memories,
            self._format_memories(memories),
        )
