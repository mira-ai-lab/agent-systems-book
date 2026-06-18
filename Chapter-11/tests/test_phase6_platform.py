"""Phase 6：entry_points、鉴权、指标。"""

import importlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_entrypoint_discovers_builtin_domains():
    from agent_framework.domain.entrypoint_loader import load_plugins_from_entrypoints
    from agent_framework.domain.plugin_registry import clear_domains, list_domains

    clear_domains()
    loaded = load_plugins_from_entrypoints()
    names = {p.name for p in loaded}
    # 书稿 conftest 或 entry_points 至少应有一个领域
    assert names or list_domains()


def test_demo_domain_registered():
    from agent_framework.domain.plugin_registry import get_domain_plugin

    plugin = get_domain_plugin("demo")
    assert plugin.display_name
    agents = plugin.create_registry().get_agent_names()
    assert "EchoAgent" in agents


def test_metrics_endpoint():
    from importlib import import_module

    api_mod = import_module("services.api.app")
    client = TestClient(api_mod.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_api_key_required_when_configured(monkeypatch):
    from importlib import import_module

    monkeypatch.setenv("API_KEYS", "test-secret")
    api_mod = import_module("services.api.app")
    mock_orch = MagicMock()
    mock_orch.process_request = AsyncMock(return_value={"final_response": "ok"})
    client = TestClient(api_mod.app)

    denied = client.post(
        "/v1/chat",
        json={"query": "hi", "domain": "demo", "user_id": "u1"},
    )
    assert denied.status_code == 401

    with patch.object(api_mod, "_get_orchestrator", AsyncMock(return_value=mock_orch)):
        ok = client.post(
            "/v1/chat",
            json={"query": "hi", "domain": "demo", "user_id": "u1"},
            headers={"X-API-Key": "test-secret"},
        )
    assert ok.status_code == 200


def test_framework_package_does_not_include_domains_in_root_pyproject():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["agent_framework*"' in text or "agent_framework*" in text
    assert "domains*" not in text.split("[tool.setuptools.packages.find]")[1].split("]")[0]
