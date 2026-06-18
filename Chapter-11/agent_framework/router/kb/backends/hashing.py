"""Hashing embedding（离线、无 API 依赖）。"""

from __future__ import annotations

from typing import List

import numpy as np

_EMBED_DIM = 384


def _tokenize(text: str) -> List[str]:
    try:
        import jieba

        return [t.strip().lower() for t in jieba.lcut(text) if t.strip()]
    except Exception:
        return [t for t in text.lower().split() if t.strip()]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


class HashingEmbeddingBackend:
    name = "hashing"

    def __init__(self, *, dim: int = _EMBED_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            vec[hash(token) % self.dim] += 1.0
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]


def embed_text(text: str, *, dim: int = _EMBED_DIM) -> np.ndarray:
    """兼容旧调用方。"""
    return HashingEmbeddingBackend(dim=dim).embed(text)
