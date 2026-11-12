"""记忆合并：整合语义或内容相近的用户画像记忆。

更新时间：2026-11-12

合并模块在保留各自独特信息的前提下生成一条综合记忆，用于控制记忆库规模
并减少重复偏好和重复事实。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from llm import call_structured_llm


def _importance(value: Any, default: float = 0.5) -> float:
    """将模型给出的重要性值转换为浮点数，并在异常时使用默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MemoryMerger:
    """基于大模型的相似记忆合并器。"""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def merge(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        """将多条相似记忆合并为一条综合记忆。

        Args:
            memories: 待合并的记忆列表，每项至少包含 ``id``、``content``
                和 ``summary`` 字段。

        Returns:
            合并后的结构化记忆；模型不可用时返回第一条记忆的降级副本。
        """
        if not memories:
            raise ValueError("至少需要一条记忆才能执行合并")

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
            merged = call_structured_llm(self.llm, prompt)
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
            "importance": _importance(merged.get("importance")),
            "timestamp": datetime.now().isoformat(),
            "metadata": {"merged_from": [memory["id"] for memory in memories]},
            "is_merged": True,
        }
