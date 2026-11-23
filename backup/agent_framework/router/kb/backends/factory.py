"""Embedding 后端工厂。"""

from __future__ import annotations

import os

from agent_framework.router.kb.backends.base import KnowledgeEmbeddingBackend
from agent_framework.router.kb.backends.embedding import OpenAIEmbeddingBackend
from agent_framework.router.kb.backends.hashing import HashingEmbeddingBackend

_BACKENDS: dict[str, type] = {
    "hashing": HashingEmbeddingBackend,
    "embedding": OpenAIEmbeddingBackend,
}


def create_knowledge_embedding_backend(name: str | None = None) -> KnowledgeEmbeddingBackend:
    value = (name or os.getenv("KNOWLEDGE_EMBEDDING_BACKEND", "hashing")).strip().lower() or "hashing"
    if value not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"未知 knowledge embedding 后端 '{value}'，可选: {supported}")
    return _BACKENDS[value]()
