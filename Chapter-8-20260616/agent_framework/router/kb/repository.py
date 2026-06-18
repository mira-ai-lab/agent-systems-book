"""知识库持久化：data/knowledge/{domain}/ + ingest / API 读写。"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from agent_framework.config import KNOWLEDGE_DIR, KNOWLEDGE_TENANT_ISOLATION
from agent_framework.router.kb.models import KnowledgeDocument
from agent_framework.router.kb.tenant import is_shared_kb_tenant, normalize_kb_tenant_id

KnowledgeStorageMode = Literal["auto", "memory", "chroma"]


def knowledge_domain_dir(domain: str) -> Path:
    return KNOWLEDGE_DIR / (domain or "").strip()


def knowledge_storage_root(domain: str, tenant_id: str = "default") -> Path:
    dom = (domain or "").strip()
    if not dom:
        raise ValueError("domain 不能为空")
    tid = normalize_kb_tenant_id(tenant_id)
    if is_shared_kb_tenant(tid) or not KNOWLEDGE_TENANT_ISOLATION:
        return knowledge_domain_dir(dom)
    return knowledge_domain_dir(dom) / "tenants" / tid


def persisted_documents_path(domain: str, tenant_id: str = "default") -> Path:
    return knowledge_storage_root(domain, tenant_id) / "documents.json"


def tenant_persisted_documents_path(domain: str, tenant_id: str) -> Path:
    tid = normalize_kb_tenant_id(tenant_id)
    if is_shared_kb_tenant(tid):
        return persisted_documents_path(domain, tid)
    return knowledge_domain_dir(domain) / "tenants" / tid / "documents.json"


def _domain_package_path(domain: str) -> Optional[Path]:
    try:
        mod = import_module(f"domains.{domain}")
        return Path(mod.__file__).resolve().parent
    except ModuleNotFoundError:
        return None


def bundle_documents_path(domain: str) -> Optional[Path]:
    root = _domain_package_path(domain)
    if root is None:
        return None
    path = root / "knowledge" / "documents.json"
    return path if path.is_file() else None


def _parse_documents_payload(payload: object) -> List[KnowledgeDocument]:
    docs_raw = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(docs_raw, list):
        return []
    documents: List[KnowledgeDocument] = []
    for item in docs_raw:
        if isinstance(item, dict):
            documents.append(KnowledgeDocument.from_dict(item))
    return documents


def load_documents_from_path(path: Path) -> List[KnowledgeDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _parse_documents_payload(payload)


def load_bundle_documents(domain: str) -> List[KnowledgeDocument]:
    path = bundle_documents_path(domain)
    if path is None:
        return []
    return load_documents_from_path(path)


def load_persisted_documents(domain: str, tenant_id: str = "default") -> List[KnowledgeDocument]:
    path = persisted_documents_path(domain, tenant_id)
    if not path.is_file():
        return []
    return load_documents_from_path(path)


def load_tenant_overlay_documents(domain: str, tenant_id: str) -> List[KnowledgeDocument]:
    tid = normalize_kb_tenant_id(tenant_id)
    if is_shared_kb_tenant(tid) or not KNOWLEDGE_TENANT_ISOLATION:
        return []
    path = tenant_persisted_documents_path(domain, tid)
    if not path.is_file():
        return []
    return load_documents_from_path(path)


def resolve_documents(
    domain: str,
    storage: KnowledgeStorageMode,
    tenant_id: str = "default",
) -> List[KnowledgeDocument]:
    dom = (domain or "").strip()
    if not dom:
        return []
    tid = normalize_kb_tenant_id(tenant_id)
    if storage == "memory":
        return load_bundle_documents(dom)
    shared = load_persisted_documents(dom, "default") or load_bundle_documents(dom)
    if is_shared_kb_tenant(tid) or not KNOWLEDGE_TENANT_ISOLATION:
        return shared
    overlay = load_tenant_overlay_documents(dom, tid)
    if not overlay:
        return shared
    return merge_documents(shared, overlay, replace=False)


def resolve_storage_mode(
    domain: str,
    storage: KnowledgeStorageMode,
    tenant_id: str = "default",
) -> str:
    mode = (storage or "auto").strip().lower() or "auto"
    if mode == "memory":
        return "memory"
    if mode == "chroma":
        return "chroma"
    tid = normalize_kb_tenant_id(tenant_id)
    if persisted_documents_path(domain, tid).is_file():
        return "chroma"
    if not is_shared_kb_tenant(tid) and tenant_persisted_documents_path(domain, tid).is_file():
        return "chroma"
    if persisted_documents_path(domain, "default").is_file():
        return "chroma"
    return "memory"


def save_persisted_documents(
    domain: str,
    documents: Sequence[KnowledgeDocument],
    *,
    tenant_id: str = "default",
) -> Path:
    dom = (domain or "").strip()
    if not dom:
        raise ValueError("domain 不能为空")
    target_dir = knowledge_storage_root(dom, tenant_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = persisted_documents_path(dom, tenant_id)
    payload = {"documents": [doc.to_dict() for doc in documents]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def merge_documents(
    existing: Sequence[KnowledgeDocument],
    incoming: Sequence[KnowledgeDocument],
    *,
    replace: bool,
) -> List[KnowledgeDocument]:
    if replace:
        return list(incoming)
    by_id = {doc.doc_id: doc for doc in existing}
    for doc in incoming:
        by_id[doc.doc_id] = doc
    return list(by_id.values())


def ingest_domain_knowledge(
    domain: str,
    *,
    embedding_backend: str = "hashing",
    source: str = "bundle",
) -> int:
    """从 bundle 或当前 persisted 写入 data/knowledge 并同步 Chroma。"""
    from agent_framework.router.kb.chroma_store import ChromaDomainKnowledgeStore
    from agent_framework.router.kb.loader import reset_domain_knowledge_cache

    dom = (domain or "").strip()
    src = (source or "bundle").strip().lower()
    if src == "bundle":
        documents = load_bundle_documents(dom)
    elif src == "persisted":
        documents = load_persisted_documents(dom)
    else:
        raise ValueError("source 可选: bundle, persisted")
    if not documents:
        raise ValueError(f"领域 '{dom}' 没有可 ingest 的知识文档")

    save_persisted_documents(dom, documents)
    store = ChromaDomainKnowledgeStore.open(dom, embedding_backend)
    store.sync_documents(documents, replace=True)
    reset_domain_knowledge_cache()
    return len(documents)


def upsert_domain_knowledge(
    domain: str,
    documents: Sequence[KnowledgeDocument],
    *,
    embedding_backend: str = "hashing",
    replace: bool = False,
    tenant_id: str = "default",
) -> int:
    from agent_framework.router.kb.chroma_store import ChromaDomainKnowledgeStore
    from agent_framework.router.kb.loader import reset_domain_knowledge_cache

    dom = (domain or "").strip()
    tid = normalize_kb_tenant_id(tenant_id)
    use_shared = is_shared_kb_tenant(tid) or not KNOWLEDGE_TENANT_ISOLATION

    if use_shared:
        base = load_persisted_documents(dom, "default") or load_bundle_documents(dom)
        merged = merge_documents(base, documents, replace=replace)
        if not merged:
            raise ValueError("documents 不能为空")
        save_persisted_documents(dom, merged, tenant_id="default")
        resolved = merged
        chroma_tid = "default"
    else:
        overlay = load_tenant_overlay_documents(dom, tid)
        merged_overlay = merge_documents(overlay, documents, replace=replace)
        if not merged_overlay:
            raise ValueError("documents 不能为空")
        save_persisted_documents(dom, merged_overlay, tenant_id=tid)
        resolved = resolve_documents(dom, "auto", tenant_id=tid)
        chroma_tid = tid

    store = ChromaDomainKnowledgeStore.open(dom, embedding_backend, tenant_id=chroma_tid)
    if replace:
        store.sync_documents(resolved, replace=True)
    else:
        store.upsert_documents(documents)
        store.sync_documents(resolved, replace=True)
    reset_domain_knowledge_cache()
    return len(resolved)


def list_domain_knowledge(
    domain: str,
    *,
    embedding_backend: str = "hashing",
    tenant_id: str = "default",
) -> dict:
    from agent_framework.router.kb.loader import get_domain_knowledge_store

    tid = normalize_kb_tenant_id(tenant_id)
    storage = resolve_storage_mode(domain, "auto", tenant_id=tid)
    documents = resolve_documents(domain, "auto", tenant_id=tid)
    store = get_domain_knowledge_store(
        domain,
        embedding_backend=embedding_backend,
        storage=storage,
        tenant_id=tid,
    )
    return {
        "domain": domain,
        "tenant_id": tid,
        "storage": getattr(store, "storage", storage) if store else storage,
        "embedding_backend": embedding_backend,
        "document_count": store.document_count if store else len(documents),
        "documents": [doc.to_dict() for doc in documents],
    }
