"""pytest 会话：确保领域插件在测试前可用。"""

from __future__ import annotations

import pytest

from agent_framework.domain.plugin_registry import (
    clear_domains,
    ensure_domains_loaded,
    list_domains,
)


def _bootstrap_domains_for_tests() -> None:
    """优先 entry_points；书稿仓库未 pip install -e domains 时手动注册。"""
    from agent_framework.domain.entrypoint_loader import load_dev_fallback_plugins

    clear_domains()
    ensure_domains_loaded()
    if list_domains():
        return
    load_dev_fallback_plugins()


@pytest.fixture(scope="session", autouse=True)
def _session_domains():
    _bootstrap_domains_for_tests()
    yield
    clear_domains()
    ensure_domains_loaded()


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要真实 LLM/API 的集成测试")
