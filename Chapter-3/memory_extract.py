"""记忆抽取：将非结构化对话转换为结构化用户画像记忆。

更新时间：2026-11-12

抽取结果包含摘要、关键点、重要性和原始内容，可作为后续合并、存储和检索
的统一数据对象。当大模型调用或结构化解析失败时，模块直接抛出
``MemoryExtractionError``，不会生成降级记忆。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from llm import call_structured_llm


class MemoryExtractionError(RuntimeError):
    """表示大模型记忆抽取未能成功完成。"""


def _importance(value: Any, default: float = 0.5) -> float:
    """将模型给出的重要性值转换为浮点数，并在异常时使用默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MemoryExtractor:
    """基于大模型的记忆抽取器。"""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def extract(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """从原始文本中抽取摘要、关键点和重要性。

        Args:
            content: 需要抽取的原始文本。
            metadata: 随记忆保存的附加元数据，例如用户标识和记忆类型。

        Returns:
            可直接交给合并或存储模块使用的结构化记忆。

        Raises:
            MemoryExtractionError: 大模型调用或 JSON 解析失败。
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

        try:
            extracted = call_structured_llm(self.llm, prompt)
        except Exception as exc:
            raise MemoryExtractionError(f"记忆抽取失败: {exc}") from exc

        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": str(extracted.get("summary", content)),
            "key_points": list(extracted.get("key_points") or []),
            "importance": _importance(extracted.get("importance")),
            "extraction_method": "llm",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def extract_interaction(
        self,
        interaction: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """从一轮包含用户和助手消息的对话中抽取长期记忆。"""
        text = "\n".join(f"{message['role']}: {message['content']}" for message in interaction)
        return self.extract(text, metadata=metadata)
