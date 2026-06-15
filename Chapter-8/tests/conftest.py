"""pytest 全局 fixture。"""

import pytest

from agent_framework.tracing import shutdown_tracing


@pytest.fixture(scope="session", autouse=True)
def _flush_tracing_after_tests():
    yield
    shutdown_tracing()
