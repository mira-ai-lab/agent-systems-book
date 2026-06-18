"""Phase 2：依赖解析、重试、并发、checkpoint 测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.domain.parsing import parse_dependency_analysis
from agent_framework.infra.checkpoint_factory import resolve_checkpoint_backend, resolve_checkpointer
from agent_framework.infra.concurrency import (
    RequestSlotTimeoutError,
    acquire_request_slot,
    max_concurrent_requests,
    reset_request_semaphore_for_tests,
)
from agent_framework.infra.resilience.retry import async_retry, is_retryable_llm_error


def test_parse_dependency_analysis_extended():
    data = {
        "order": {"1": "T1", "2": "T2", "3": "T3"},
        "depends_on": {"T2": ["T1"], "T3": ["T1", "T2"]},
    }
    order, deps = parse_dependency_analysis(data, ["T1", "T2", "T3"])
    assert order == ["T1", "T2", "T3"]
    assert deps["T2"] == ["T1"]
    assert deps["T3"] == ["T1", "T2"]


def test_parse_dependency_analysis_legacy():
    data = {"1": "T2", "2": "T1"}
    order, deps = parse_dependency_analysis(data, ["T1", "T2"])
    assert order == ["T2", "T1"]
    assert deps == {"T1": [], "T2": []}


def test_is_retryable_llm_error():
    assert is_retryable_llm_error(Exception("429 rate limit exceeded"))
    assert not is_retryable_llm_error(ValueError("bad json"))


def test_async_retry_eventually_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("timeout connecting")
        return "ok"

    result = asyncio.run(async_retry(flaky, max_attempts=3, base_delay_sec=0.01))
    assert result == "ok"
    assert calls["n"] == 2


def test_resolve_checkpoint_memory():
    assert resolve_checkpoint_backend("memory") == "memory"
    cp = resolve_checkpointer("memory")
    assert cp is not None


def test_resolve_checkpoint_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "cp.db"
    monkeypatch.setenv("CHECKPOINT_SQLITE_PATH", str(db))
    cp = resolve_checkpointer("sqlite")
    assert cp is not None


def test_request_slot_timeout(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "1")
    reset_request_semaphore_for_tests()

    async def _run():
        async with acquire_request_slot():
            with pytest.raises(RequestSlotTimeoutError):
                async with acquire_request_slot(wait_timeout_sec=0.05):
                    pass

    asyncio.run(_run())


def test_topological_layers_with_deps():
    from agent_framework.orchestration.fixed_graph.nodes import _topological_layers

    plan = {
        "execution_order": ["T1", "T2", "T3"],
        "subtasks": [
            {"task_id": "T1", "depends_on": []},
            {"task_id": "T2", "depends_on": ["T1"]},
            {"task_id": "T3", "depends_on": ["T1", "T2"]},
        ],
    }
    layers = _topological_layers(plan)
    assert layers == [["T1"], ["T2"], ["T3"]]
