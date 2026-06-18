"""加载 domains/{domain}/knowledge 或 data/knowledge/{domain}/。"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Union

from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend
from agent_framework.router.kb.repository import (
    KnowledgeStorageMode,
    resolve_documents,
    resolve_storage_mode,
)
from agent_framework.router.kb.tenant import normalize_kb_tenant_id
from agent_framework.router.kb.vector_store import DomainKnowledgeStore

KnowledgeStore = Union[DomainKnowledgeStore, "ChromaDomainKnowledgeStore"]


@lru_cache(maxsize=64)
def get_domain_knowledge_store(
    domain: str,
    embedding_backend: str = "hashing",
    storage: KnowledgeStorageMode = "auto",
    tenant_id: str = "default",
) -> Optional[KnowledgeStore]:
    dom = (domain or "").strip()
    if not dom:
        return None
    tid = normalize_kb_tenant_id(tenant_id)
    mode = resolve_storage_mode(dom, storage, tenant_id=tid)
    documents = resolve_documents(dom, storage, tenant_id=tid)
    if not documents:
        return None

    if mode == "chroma":
        from agent_framework.router.kb.chroma_store import ChromaDomainKnowledgeStore

        return ChromaDomainKnowledgeStore.open(dom, embedding_backend, tenant_id=tid)

    backend = create_knowledge_embedding_backend(embedding_backend)
    return DomainKnowledgeStore(documents, embedding_backend=backend)


def reset_domain_knowledge_cache() -> None:
    get_domain_knowledge_store.cache_clear()
