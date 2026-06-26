"""Phase 25 P0：多租户 KB 隔离 + 增量 upsert。"""

import pytest

from agent_framework.router.kb.loader import get_domain_knowledge_store, reset_domain_knowledge_cache
from agent_framework.router.kb.models import KnowledgeDocument
from agent_framework.router.kb.repository import (
    list_domain_knowledge,
    resolve_documents,
    upsert_domain_knowledge,
)
from agent_framework.router.kb.tenant import normalize_kb_tenant_id
from agent_framework.router.stages.knowledge_routing import match_vector_knowledge_candidates


@pytest.fixture(autouse=True)
def _reset_kb_cache():
    reset_domain_knowledge_cache()
    yield
    reset_domain_knowledge_cache()


def test_tenant_overlay_merges_with_shared(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    from agent_framework.router.kb.repository import ingest_domain_knowledge

    ingest_domain_knowledge("travel", embedding_backend="hashing")
    upsert_domain_knowledge(
        "travel",
        [
            KnowledgeDocument(
                doc_id="tenant-vip-faq",
                agent="WeatherAgent",
                text="VIP 客户专享极速退款通道。",
                tags=["VIP"],
            )
        ],
        tenant_id="alice",
    )
    merged = resolve_documents("travel", "auto", tenant_id="alice")
    ids = {doc.doc_id for doc in merged}
    assert "tenant-vip-faq" in ids
    assert len(merged) >= 4


def test_tenant_kb_isolated_from_other_tenants(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    from agent_framework.router.kb.repository import ingest_domain_knowledge

    ingest_domain_knowledge("travel", embedding_backend="hashing")
    upsert_domain_knowledge(
        "travel",
        [
            KnowledgeDocument(
                doc_id="alice-only",
                agent="WeatherAgent",
                text="alice 私有知识。",
                tags=["private"],
            )
        ],
        tenant_id="alice",
    )
    bob_docs = resolve_documents("travel", "auto", tenant_id="bob")
    assert all(doc.doc_id != "alice-only" for doc in bob_docs)


def test_tenant_vector_routing_uses_overlay(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    from agent_framework.router.kb.repository import ingest_domain_knowledge

    ingest_domain_knowledge("travel", embedding_backend="hashing")
    upsert_domain_knowledge(
        "travel",
        [
            KnowledgeDocument(
                doc_id="tenant-route-doc",
                agent="WeatherAgent",
                text="专属退货绿色通道政策说明。",
                tags=["退货"],
            )
        ],
        tenant_id="alice",
    )
    hits, meta = match_vector_knowledge_candidates(
        "travel",
        query="专属退货绿色通道",
        embedding_backend="hashing",
        storage="chroma",
        tenant_id="alice",
        min_score=0.01,
        vector_min_score=0.01,
    )
    assert hits
    assert any(item.get("doc_id") == "tenant-route-doc" for item in meta)


def test_incremental_upsert_preserves_existing_docs(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    from agent_framework.router.kb.repository import ingest_domain_knowledge

    ingest_domain_knowledge("travel", embedding_backend="hashing")
    before = list_domain_knowledge("travel")
    before_count = before["document_count"]
    upsert_domain_knowledge(
        "travel",
        [
            KnowledgeDocument(
                doc_id="incremental-doc",
                agent="WeatherAgent",
                text="增量写入文档。",
                tags=["增量"],
            )
        ],
        replace=False,
    )
    after = list_domain_knowledge("travel")
    assert after["document_count"] == before_count + 1


def test_api_knowledge_tenant_scope(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import agent_framework.router.kb.repository as repo
    from importlib import import_module

    from fastapi.testclient import TestClient

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    from agent_framework.router.kb.repository import ingest_domain_knowledge

    ingest_domain_knowledge("travel", embedding_backend="hashing")
    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)

    posted = client.post(
        "/v1/domains/travel/knowledge",
        params={"user_id": "alice"},
        json={
            "documents": [
                {
                    "id": "api-tenant-doc",
                    "agent": "WeatherAgent",
                    "text": "alice API 租户文档。",
                    "tags": ["tenant"],
                }
            ]
        },
    )
    assert posted.status_code == 200
    assert posted.json()["tenant_id"] == "alice"

    listed = client.get(
        "/v1/domains/travel/knowledge",
        params={"user_id": "alice"},
    )
    assert listed.status_code == 200
    assert any(doc["id"] == "api-tenant-doc" for doc in listed.json()["documents"])


def test_normalize_kb_tenant_id():
    assert normalize_kb_tenant_id(None) == "default"
    assert normalize_kb_tenant_id("  bob  ") == "bob"
