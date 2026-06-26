"""Phase 25 P1：Registry 联邦（25.5–25.8）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.domain.registry_federation import (
    federation_cluster_name,
    list_federated_agents,
    normalize_federated_agent,
    parse_federation_urls,
    probe_a2a_endpoint,
    probe_cluster_health,
)


def test_parse_federation_urls():
    assert parse_federation_urls("http://a:8780/, http://b:8780") == [
        "http://a:8780",
        "http://b:8780",
    ]
    assert parse_federation_urls("") == []


def test_federation_cluster_name():
    assert federation_cluster_name("http://127.0.0.1:8780") == "127.0.0.1:8780"


def test_normalize_federated_agent():
    entry = normalize_federated_agent(
        {"name": "RemoteFAQ", "domain": "travel", "source": "static"},
        cluster="remote-a",
        base_url="http://remote-a:8780",
    )
    assert entry["origin"] == "federated"
    assert entry["federation_cluster"] == "remote-a"
    assert entry["name"] == "RemoteFAQ"


def test_list_federated_agents_merges_local_and_remote():
    remote_payload = {
        "agents": [
            {
                "name": "RemoteFAQ",
                "domain": "travel",
                "source": "static",
                "description": "remote faq",
            }
        ],
        "count": 1,
    }

    async def _run():
        with patch(
            "agent_framework.domain.registry_federation.fetch_remote_registry",
            AsyncMock(return_value=remote_payload),
        ):
            return await list_federated_agents(
                federation_urls=["http://remote-a:8780"],
            )

    result = asyncio.run(_run())

    origins = {item["origin"] for item in result["agents"]}
    assert "local" in origins
    assert "federated" in origins
    assert result["federated_count"] == 1
    assert result["federation"][0]["status"] == "ok"
    assert any(a["name"] == "RemoteFAQ" for a in result["agents"])


def test_list_federated_agents_records_fetch_error():
    async def _run():
        with patch(
            "agent_framework.domain.registry_federation.fetch_remote_registry",
            AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            return await list_federated_agents(federation_urls=["http://down:8780"])

    result = asyncio.run(_run())

    assert result["federated_count"] == 0
    assert result["federation"][0]["status"] == "error"
    assert "connection refused" in result["federation"][0]["error"]


def test_probe_a2a_endpoint_success():
    mock_response = MagicMock()
    mock_response.status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            if url.endswith("/.well-known/agent.json"):
                return mock_response
            raise RuntimeError("not found")

    async def _run():
        with patch("agent_framework.domain.registry_federation.httpx.AsyncClient", lambda **kwargs: FakeClient()):
            return await probe_a2a_endpoint("http://127.0.0.1:9012")

    health = asyncio.run(_run())

    assert health["reachable"] is True
    assert health["status_code"] == 200


def test_probe_cluster_health():
    class Resp:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            if str(url).endswith("/health"):
                return Resp()
            raise RuntimeError("skip")

    async def _run():
        with patch("agent_framework.domain.registry_federation.httpx.AsyncClient", lambda **kwargs: FakeClient()):
            return await probe_cluster_health("http://127.0.0.1:8780")

    health = asyncio.run(_run())
    assert health["reachable"] is True


def test_list_federated_agents_attaches_a2a_health():
    remote_payload = {
        "agents": [
            {
                "name": "RemoteA2A",
                "domain": "demo",
                "source": "dynamic",
                "a2a_url": "http://127.0.0.1:9012",
            }
        ]
    }

    async def _run():
        with patch(
            "agent_framework.domain.registry_federation.fetch_remote_registry",
            AsyncMock(return_value=remote_payload),
        ), patch(
            "agent_framework.domain.registry_federation.probe_a2a_endpoint",
            AsyncMock(return_value={"reachable": True, "status_code": 200}),
        ), patch(
            "agent_framework.domain.registry_federation.probe_cluster_health",
            AsyncMock(return_value={"reachable": True, "status_code": 200}),
        ):
            return await list_federated_agents(
                federation_urls=["http://remote:8780"],
                include_health=True,
            )

    result = asyncio.run(_run())

    remote = next(a for a in result["agents"] if a.get("name") == "RemoteA2A")
    assert remote["a2a_health"]["reachable"] is True
    assert result["federation"][0]["health"]["reachable"] is True


def test_api_agents_federated_endpoint():
    pytest.importorskip("fastapi")
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)

    with patch(
        "agent_framework.domain.registry_federation.list_federated_agents",
        AsyncMock(
            return_value={
                "agents": [{"name": "RemoteFAQ", "origin": "federated"}],
                "count": 1,
                "local_count": 0,
                "federated_count": 1,
                "federation": [{"cluster": "remote", "status": "ok", "count": 1}],
                "filters": {"federated": True, "health": False},
            }
        ),
    ):
        resp = client.get("/v1/agents", params={"federated": "true"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["federated_count"] == 1
    assert body["filters"]["federated"] is True


def test_api_registry_federation_status():
    pytest.importorskip("fastapi")
    from importlib import import_module

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)

    with patch(
        "agent_framework.domain.registry_federation.parse_federation_urls",
        return_value=["http://remote:8780"],
    ), patch(
        "agent_framework.domain.registry_federation.list_federation_clusters",
        AsyncMock(
            return_value=[
                {
                    "cluster": "remote:8780",
                    "base_url": "http://remote:8780",
                    "status": "configured",
                    "health": {"reachable": True},
                }
            ]
        ),
    ):
        resp = client.get("/v1/registry/federation")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster_count"] == 1
    assert body["clusters"][0]["health"]["reachable"] is True
