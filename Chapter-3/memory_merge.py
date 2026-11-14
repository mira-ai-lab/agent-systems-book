"""Memory merging: consolidate semantically or textually similar memories.

The merger creates one consolidated memory while preserving each source
memory's unique information. This controls memory size and reduces duplicate
preferences and facts.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from llm import chat


class MemoryMerger:
    """Merge similar memories with an instruction-following LLM."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

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

        memories_text = "\n\n".join(
            f"记忆{index + 1}:\n{memory['content']}"
            for index, memory in enumerate(memories)
        )
        prompt = f"""请将以下多个相似记忆合并为一个综合记忆，保留所有独特的重要信息：
        {memories_text}

        只输出一个 JSON 对象：
        {{
            "summary": "合并后的一句话总结",
            "key_points": ["合并后的要点1", "合并后的要点2"],
            "importance": 0.5
        }}"""

        try:
            merged = chat(self.llm, prompt)
        except Exception:
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
