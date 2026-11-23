"""知识库 Embedding 后端。"""

from agent_framework.router.kb.backends.base import KnowledgeEmbeddingBackend
from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend

__all__ = ["KnowledgeEmbeddingBackend", "create_knowledge_embedding_backend"]
