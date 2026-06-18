"""KnowledgeEmbeddingBackend 协议。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class KnowledgeEmbeddingBackend(Protocol):
    """将文本映射为单位向量，供 cosine 相似度检索。"""

    name: str

    def embed(self, text: str) -> np.ndarray: ...

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]
