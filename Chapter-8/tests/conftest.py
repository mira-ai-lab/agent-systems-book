"""Shared pytest fixtures.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

import pytest

from travel_multi_agent.tracing import shutdown_tracing


@pytest.fixture(scope="session", autouse=True)
def _flush_tracing_after_tests():
    yield
    shutdown_tracing()
