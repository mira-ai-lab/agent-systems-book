"""Chroma 持久化知识库（与 chroma_memory 长期记忆隔离）。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from agent_framework.router.kb.backends.base import KnowledgeEmbeddingBackend
from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend
from agent_framework.router.kb.models import KnowledgeDocument
from agent_framework.router.kb.repository import knowledge_storage_root, resolve_documents
from agent_framework.router.kb.tenant import normalize_kb_tenant_id


class ChromaDomainKnowledgeStore:
    """data/knowledge/{domain}/chroma 上的向量索引。"""

    storage = "chroma"

    def __init__(
        self,
        domain: str,
        collection,
        *,
        embedding_backend: KnowledgeEmbeddingBackend,
        documents: Sequence[KnowledgeDocument],
    ) -> None:
        self.domain = domain
        self._collection = collection
        self.backend = embedding_backend
        self.documents = list(documents)

    @property
    def embedding_backend_name(self) -> str:
        return self.backend.name

    @property
    def document_count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return len(self.documents)

    @classmethod
    def open(
        cls,
        domain: str,
        embedding_backend_name: str,
        *,
        tenant_id: str = "default",
    ) -> "ChromaDomainKnowledgeStore":
        import chromadb

        tid = normalize_kb_tenant_id(tenant_id)
        backend = create_knowledge_embedding_backend(embedding_backend_name)
        chroma_path = knowledge_storage_root(domain, tid) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name=f"kb_{embedding_backend_name}",
            metadata={"hnsw:space": "cosine", "domain": domain, "tenant_id": tid},
        )
        documents = resolve_documents(domain, "auto", tenant_id=tid)
        store = cls(domain, collection, embedding_backend=backend, documents=documents)
        if documents and store.document_count == 0:
            store.sync_documents(documents, replace=True)
        return store

    def upsert_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        """增量 upsert：仅写入/更新指定 doc_id，不删除全库。"""
        docs = list(documents)
        if not docs:
            return
        ids = [doc.doc_id for doc in docs]
        texts = [doc.text for doc in docs]
        embeddings = [self.backend.embed(text).tolist() for text in texts]
        metadatas = [
            {
                "agent": doc.agent,
                "doc_id": doc.doc_id,
                "tags": ",".join(doc.tags),
            }
            for doc in docs
        ]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        by_id = {doc.doc_id: doc for doc in self.documents}
        for doc in docs:
            by_id[doc.doc_id] = doc
        self.documents = list(by_id.values())

    def sync_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        replace: bool = False,
    ) -> None:
        docs = list(documents)
        if replace and self.document_count > 0:
            existing = self._collection.get(include=[])
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)

        if not docs:
            self.documents = []
            return

        ids = [doc.doc_id for doc in docs]
        texts = [doc.text for doc in docs]
        embeddings = [self.backend.embed(text).tolist() for text in texts]
        metadatas = [
            {
                "agent": doc.agent,
                "doc_id": doc.doc_id,
                "tags": ",".join(doc.tags),
            }
            for doc in docs
        ]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.documents = docs

    def search(
        self,
        query: str,
        events: Sequence[str] = (),
        *,
        top_k: int = 5,
        min_score: float = 0.65,
    ) -> List[Tuple[str, float, str]]:
        if self.document_count == 0:
            return []
        haystack = " ".join([query.strip(), *(str(e).strip() for e in events if str(e).strip())])
        query_vec = self.backend.embed(haystack)
        n_results = min(max(top_k * 3, top_k), self.document_count)
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=max(1, n_results),
            include=["metadatas", "distances"],
        )
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        scored: List[Tuple[str, float, str]] = []
        for meta, distance in zip(metas, distances):
            if not meta:
                continue
            raw_score = 1.0 - float(distance)
            if raw_score < min_score:
                continue
            scored.append((str(meta["agent"]), raw_score, str(meta["doc_id"])))
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
