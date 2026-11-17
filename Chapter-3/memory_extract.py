"""Memory extraction: convert unstructured dialogue into user-profile memories.

The extraction result contains a summary, key points, importance, and the
original content, providing a consistent data object for later merging,
storage, and retrieval.

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


class MemoryExtractor:
    """Extract structured memories with an instruction-following LLM."""

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
    def _build_prompt(content: str) -> str:
        return f"""请从以下内容中提取用户画像和偏好记忆，保留所有重要信息：
        {content}

        只输出一个 JSON 对象，字段如下：
        {{
            "summary": "一句话总结",
            "key_points": ["要点1", "要点2"],
            "importance": 0.5
        }}
        importance 为 0.0 到 1.0 之间的数字。"""

    @staticmethod
    def _build_result(
        content: str,
        extracted: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": str(extracted.get("summary", content)),
            "key_points": list(extracted.get("key_points") or []),
            "importance": float(extracted.get("importance", 0.5)),
            "extraction_method": "llm",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def extract(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract a summary, key points, and importance from raw text.

        Args:
            content: Raw text to extract memory from.
            metadata: Additional metadata to persist with the memory, such as
                the user ID and memory type.

        Returns:
            A structured memory ready for the merge or storage modules.
        """
        prompt = self._build_prompt(content)
        extracted = (self.chat_fn or _default_chat_fn)(self.llm, prompt)
        return self._build_result(content, extracted, metadata)

    async def extract_async(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously extract a structured memory.

        Args:
            content: Raw text to extract memory from.
            metadata: Additional metadata to persist with the memory.

        Returns:
            A structured memory ready for merging and storage.
        """
        prompt = self._build_prompt(content)
        if self.async_chat_fn:
            extracted = await self.async_chat_fn(self.llm, prompt)
        else:
            chat_fn = self.chat_fn or _default_chat_fn
            extracted = await asyncio.to_thread(chat_fn, self.llm, prompt)
        return self._build_result(content, extracted, metadata)
