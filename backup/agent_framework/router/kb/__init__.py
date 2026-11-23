"""Router 向量知识库（per-domain documents → agent 候选）。"""

from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend
from agent_framework.router.kb.loader import get_domain_knowledge_store, reset_domain_knowledge_cache
from agent_framework.router.kb.models import KnowledgeDocument
from agent_framework.router.kb.repository import ingest_domain_knowledge, list_domain_knowledge, upsert_domain_knowledge
from agent_framework.router.kb.scoring import attach_normalized_scores, normalize_keyword_score, normalize_vector_score
from agent_framework.router.kb.vector_store import DomainKnowledgeStore

__all__ = [
    "DomainKnowledgeStore",
    "KnowledgeDocument",
    "attach_normalized_scores",
    "create_knowledge_embedding_backend",
    "get_domain_knowledge_store",
    "ingest_domain_knowledge",
    "list_domain_knowledge",
    "normalize_keyword_score",
    "normalize_vector_score",
    "reset_domain_knowledge_cache",
    "upsert_domain_knowledge",
]
