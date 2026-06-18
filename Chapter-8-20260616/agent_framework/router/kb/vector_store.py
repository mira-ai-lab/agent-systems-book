"""轻量向量知识库：可插拔 Embedding 后端 + cosine 相似度。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from agent_framework.router.kb.backends.base import KnowledgeEmbeddingBackend
from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend
from agent_framework.router.kb.models import KnowledgeDocument


class DomainKnowledgeStore:
    """内存向量索引；文档来自 domains/{domain}/knowledge/documents.json。"""

    storage = "memory"

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        embedding_backend: Optional[KnowledgeEmbeddingBackend] = None,
    ) -> None:
        self.documents = list(documents)
        self.backend: KnowledgeEmbeddingBackend = embedding_backend or create_knowledge_embedding_backend(
            "hashing"
        )
        self._vectors = self.backend.embed_documents([doc.text for doc in self.documents])

    @property
    def embedding_backend_name(self) -> str:
        return self.backend.name

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def search(
        self,
        query: str,
        events: Sequence[str] = (),
        *,
        top_k: int = 5,
        min_score: float = 0.65,
    ) -> List[Tuple[str, float, str]]:
        """返回 (agent, score, doc_id) 列表。"""
        if not self.documents:
            return []
        haystack = " ".join([query.strip(), *(str(e).strip() for e in events if str(e).strip())])
        query_vec = self.backend.embed(haystack)
        scored: List[Tuple[str, float, str]] = []
        for doc, doc_vec in zip(self.documents, self._vectors):
            score = float(np.dot(query_vec, doc_vec))
            if score >= min_score:
                scored.append((doc.agent, score, doc.doc_id))
        scored.sort(key=lambda item: -item[1])
        return scored[: max(1, top_k)]

    def match_agents(
        self,
        query: str,
        events: Sequence[str] = (),
        *,
        top_k: int = 5,
        min_score: float = 0.65,
    ) -> List[Tuple[str, float, str]]:
        hits = self.search(query, events, top_k=top_k, min_score=min_score)
        best_by_agent: dict[str, Tuple[float, str]] = {}
        for agent, score, doc_id in hits:
            prev = best_by_agent.get(agent)
            if prev is None or score > prev[0]:
                best_by_agent[agent] = (score, doc_id)
        merged = [(agent, score, doc_id) for agent, (score, doc_id) in best_by_agent.items()]
        merged.sort(key=lambda item: -item[1])
        return merged
