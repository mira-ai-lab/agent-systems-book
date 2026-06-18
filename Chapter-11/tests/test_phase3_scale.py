"""Phase 3：多租户、异步任务、Docker 相关单元测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.bootstrap.tenant_pool import TenantOrchestratorPool
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH
from services.jobs.models import JobStatus
from services.jobs.store import JobStore


def test_job_store_lifecycle(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create_job(user_id="u1", query="规划上海三日游", thread_id="t1", domain="travel")
    assert job.status == JobStatus.PENDING

    claimed = store.claim_next_pending()
    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == JobStatus.RUNNING

    store.mark_succeeded(job.job_id, {"final_response": "行程已生成"}, trace_id="tr-1")
    done = store.get_job(job.job_id)
    assert done is not None
    assert done.status == JobStatus.SUCCEEDED
    assert done.trace_id == "tr-1"
    payload = done.to_dict()
    assert payload["result"]["final_response"] == "行程已生成"


def test_job_store_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create_job(user_id="u2", query="test", thread_id="t2", domain="travel")
    store.claim_next_pending()
    store.mark_failed(job.job_id, "boom")
    failed = store.get_job(job.job_id)
    assert failed is not None
    assert failed.status == JobStatus.FAILED
    assert failed.error == "boom"


def test_tenant_pool_caches_by_user_id():
    pool = TenantOrchestratorPool(max_size=4)
    created = []

    def fake_create(domain, mode=MODE_FIXED_GRAPH, **kwargs):
        runtime = MagicMock()
        runtime.user_id = kwargs.get("user_id")
        runtime.domain = domain
        runtime.mode = mode
        created.append((domain, mode, kwargs.get("user_id")))
        return runtime

    with patch("agent_framework.bootstrap.platform.create_runtime", side_effect=fake_create):
        o1 = asyncio.run(pool.get("alice", domain="travel", mode="fixed_graph"))
        o2 = asyncio.run(pool.get("alice", domain="travel", mode="fixed_graph"))
        o3 = asyncio.run(pool.get("alice", domain="travel", mode="supervisor"))

    assert o1 is o2
    assert o1 is not o3
    assert created == [
        ("travel", "fixed_graph", "alice"),
        ("travel", "supervisor", "alice"),
    ]


@pytest.mark.parametrize("path", ["health", "ready"])
def test_api_health_ready(path):
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    mock_orch = MagicMock()
    with patch.object(api_mod, "_get_orchestrator", AsyncMock(return_value=mock_orch)):
        client = TestClient(api_mod.app)
        resp = client.get(f"/{path}")
        assert resp.status_code == 200


def test_api_submit_and_get_job():
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    submit = client.post(
        "/v1/jobs",
        json={"query": "杭州三日游", "user_id": "u99", "domain": "travel"},
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]
    detail = client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "pending"
    assert detail.json()["user_id"] == "u99"
