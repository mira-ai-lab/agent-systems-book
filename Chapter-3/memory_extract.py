"""Memory extraction: convert unstructured dialogue into user-profile memories.

The extraction result contains a summary, key points, importance, and the
original content, providing a consistent data object for later merging,
storage, and retrieval.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from llm import chat


class MemoryExtractor:
    """Extract structured memories with an instruction-following LLM."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

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
        prompt = f"""请从以下内容中提取用户画像和偏好记忆，保留所有重要信息：
        {content}

        只输出一个 JSON 对象，字段如下：
        {{
            "summary": "一句话总结",
            "key_points": ["要点1", "要点2"],
            "importance": 0.5
        }}
        importance 为 0.0 到 1.0 之间的数字。"""

        extracted = chat(self.llm, prompt)

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
