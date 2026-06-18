"""Phase 24：route(query) + KB Embedding + HTTP 对齐 + 可观测字段。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agent_framework.router.kb.backends.factory import create_knowledge_embedding_backend
from agent_framework.router.kb.backends.hashing import HashingEmbeddingBackend
from agent_framework.router.kb.loader import get_domain_knowledge_store, reset_domain_knowledge_cache
from agent_framework.router.kb.repository import ingest_domain_knowledge, list_domain_knowledge, upsert_domain_knowledge
from agent_framework.router.kb.scoring import normalize_keyword_score, normalize_vector_score
from agent_framework.router.observability import enrich_routing_observability, knowledge_matches_from_result
from agent_framework.router.stages.knowledge_routing import match_vector_knowledge_candidates, resolve_knowledge_candidates
from agent_framework.router.config import RouterConfig
from agent_framework.domain.plugin_registry import get_domain_plugin


@pytest.fixture(autouse=True)
def _reset_kb_cache():
    reset_domain_knowledge_cache()
    yield
    reset_domain_knowledge_cache()


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    reset_domain_knowledge_cache()
    yield tmp_path
    reset_domain_knowledge_cache()


def test_create_knowledge_embedding_backend_hashing():
    backend = create_knowledge_embedding_backend("hashing")
    assert isinstance(backend, HashingEmbeddingBackend)
    vec = backend.embed("退换货政策")
    assert vec.shape[0] > 0
    assert pytest.approx(1.0, rel=1e-3) == float(np.linalg.norm(vec))


def test_create_knowledge_embedding_backend_unknown():
    with pytest.raises(ValueError, match="未知 knowledge embedding"):
        create_knowledge_embedding_backend("unknown")


def test_domain_knowledge_store_uses_embedding_backend_param():
    store_a = get_domain_knowledge_store("customer_service", embedding_backend="hashing")
    store_b = get_domain_knowledge_store("customer_service", embedding_backend="hashing")
    assert store_a is store_b
    assert store_a is not None
    assert store_a.embedding_backend_name == "hashing"


def test_match_vector_includes_embedding_backend_meta(isolated_kb):
    candidates, meta = match_vector_knowledge_candidates(
        "customer_service",
        query="退换货政策",
        min_score=0.15,
        embedding_backend="hashing",
    )
    assert candidates
    assert meta[0]["embedding_backend"] == "hashing"
    assert "raw_score" in meta[0]
    assert "normalized_score" in meta[0]
    assert 0.0 <= meta[0]["normalized_score"] <= 1.0


def test_route_sdk_entry(monkeypatch):
    import agent_framework.bootstrap.entry as entry_module

    mock_runtime = MagicMock()
    mock_runtime.process_request = AsyncMock(
        return_value={"final_response": "ok", "resolved_profile": "adaptive"}
    )
    monkeypatch.setattr(entry_module, "create_runtime", lambda *a, **kw: mock_runtime)

    result = asyncio.run(entry_module.route("退货政策是什么？", domain="customer_service"))
    assert result["domain"] == "customer_service"
    assert result["resolved_domain"] == "customer_service"
    assert result["knowledge_matches"] == []
    assert result["final_response"] == "ok"
    mock_runtime.process_request.assert_awaited_once()


def test_route_resolves_default_domain(monkeypatch):
    import agent_framework.bootstrap.entry as entry_module

    monkeypatch.setattr(entry_module, "DEFAULT_DOMAIN", "demo")
    mock_runtime = MagicMock()
    mock_runtime.process_request = AsyncMock(return_value={"final_response": "echo"})
    captured: dict = {}

    def fake_create_runtime(domain, **kwargs):
        captured["domain"] = domain
        return mock_runtime

    monkeypatch.setattr(entry_module, "create_runtime", fake_create_runtime)
    result = asyncio.run(entry_module.route("hello"))
    assert captured["domain"] == "demo"
    assert result["domain"] == "demo"
    assert result["resolved_domain"] == "demo"


def test_enrich_routing_observability_promotes_knowledge_matches():
    result = {
        "final_response": "ok",
        "resolved_profile": "workflow",
        "routing_plan": {
            "metadata": {
                "knowledge_matches": [{"name": "FAQAgent", "score": 0.9, "source": "vector"}],
            }
        },
    }
    enriched = enrich_routing_observability(result, domain="customer_service")
    assert enriched["resolved_domain"] == "customer_service"
    assert enriched["knowledge_matches"][0]["source"] == "vector"
    assert knowledge_matches_from_result(enriched)[0]["name"] == "FAQAgent"


def test_list_domains_includes_recommended_profile():
    from agent_framework.domain.plugin_registry import list_domains

    cs = next(d for d in list_domains() if d["name"] == "customer_service")
    assert cs["recommended_profile"] == "auto"


def test_api_domains_endpoint_recommended_profile():
    pytest.importorskip("fastapi")
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    resp = client.get("/v1/domains")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended_profile"] == "auto"
    assert body["domains"]
    assert all(d.get("recommended_profile") == "auto" for d in body["domains"])


def test_chat_request_openapi_example_query_only():
    pytest.importorskip("fastapi")
    from importlib import import_module

    api_mod = import_module("services.api.app")
    schema = api_mod.ChatRequest.model_json_schema()
    examples = schema.get("examples") or []
    assert examples
    assert "query" in examples[0]
    assert "domain" not in examples[0]


def test_score_normalization_keyword_and_vector():
    assert normalize_keyword_score(0.65) == pytest.approx(0.0)
    assert normalize_keyword_score(0.8) == pytest.approx(0.428571, rel=1e-3)
    assert normalize_keyword_score(1.0) == pytest.approx(1.0)
    assert normalize_vector_score(0.15) == pytest.approx(0.0)
    assert normalize_vector_score(0.575) == pytest.approx(0.5, rel=1e-3)
    assert normalize_vector_score(1.0) == pytest.approx(1.0)


def test_hybrid_knowledge_meta_has_normalized_scores():
    registry = get_domain_plugin("customer_service").create_registry()
    _, meta = resolve_knowledge_candidates(
        registry,
        domain="customer_service",
        query="咨询退货政策",
        events=[],
        config=RouterConfig(knowledge_backend="hybrid", knowledge_min_score=0.15),
    )
    assert meta
    for item in meta:
        assert "raw_score" in item
        assert "normalized_score" in item
        assert item["normalized_score"] == item["score"]


def test_ingest_knowledge_chroma_persistence(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    reset_domain_knowledge_cache()
    count = ingest_domain_knowledge("customer_service", embedding_backend="hashing")
    assert count == 3
    assert (tmp_path / "customer_service" / "documents.json").is_file()
    assert (tmp_path / "customer_service" / "chroma").is_dir()

    payload = list_domain_knowledge("customer_service", embedding_backend="hashing")
    assert payload["storage"] == "chroma"
    assert payload["document_count"] == 3

    store = get_domain_knowledge_store("customer_service", embedding_backend="hashing", storage="chroma")
    assert store is not None
    hits = store.match_agents("退换货政策", [], min_score=0.15)
    assert hits
    assert hits[0][0] == "FAQAgent"

    reset_domain_knowledge_cache()
    store_reloaded = get_domain_knowledge_store("customer_service", embedding_backend="hashing", storage="auto")
    assert store_reloaded is not None
    assert store_reloaded.storage == "chroma"
    hits2 = store_reloaded.match_agents("退换货政策", [], min_score=0.15)
    assert hits2[0][0] == "FAQAgent"


def test_upsert_domain_knowledge_via_repository(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    from agent_framework.router.kb.models import KnowledgeDocument

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    reset_domain_knowledge_cache()
    ingest_domain_knowledge("customer_service", embedding_backend="hashing")
    count = upsert_domain_knowledge(
        "customer_service",
        [
            KnowledgeDocument(
                doc_id="cs-new-faq",
                agent="FAQAgent",
                text="会员积分可在下单时抵扣。",
                tags=["积分"],
            )
        ],
        embedding_backend="hashing",
    )
    assert count == 4
    payload = list_domain_knowledge("customer_service")
    assert any(doc["id"] == "cs-new-faq" for doc in payload["documents"])


def test_api_knowledge_get_and_post(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import agent_framework.router.kb.repository as repo
    from importlib import import_module

    from fastapi.testclient import TestClient

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    reset_domain_knowledge_cache()
    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)

    listed = client.get("/v1/domains/customer_service/knowledge")
    assert listed.status_code == 200
    body = listed.json()
    assert body["domain"] == "customer_service"
    assert body["documents"]

    posted = client.post(
        "/v1/domains/customer_service/knowledge",
        json={
            "documents": [
                {
                    "id": "api-doc-1",
                    "agent": "FAQAgent",
                    "text": "API 热更新文档：支持 7 天无理由退货。",
                    "tags": ["退货"],
                }
            ],
            "replace": False,
        },
    )
    assert posted.status_code == 200
    post_body = posted.json()
    assert post_body["document_count"] >= 1
    assert post_body["storage"] == "chroma"
    assert post_body["invalidated_runtimes"] >= 0
