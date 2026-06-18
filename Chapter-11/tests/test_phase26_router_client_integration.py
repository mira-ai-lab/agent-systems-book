"""Phase 26.1：router-client SDK ↔ FastAPI 联调（uvicorn + Node integration）。"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "packages" / "router-client"


def _npm_executable() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_sec: float = 10.0) -> None:
    deadline = time.time() + timeout_sec
    url = f"{base_url}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.05)
    raise RuntimeError(f"API not ready at {url}")


@pytest.fixture
def mock_router_api_server():
    pytest.importorskip("uvicorn")
    from importlib import import_module

    import uvicorn

    api_mod = import_module("services.api.app")
    mock_orch = MagicMock()
    mock_orch.process_request = AsyncMock(
        return_value={
            "final_response": "北京明天晴，适合出行。",
            "trace_id": "trace-sdk-integration",
            "span_id": "span-sdk-integration",
        }
    )

    async def fake_stream(*args, **kwargs):
        yield {
            "type": "router.extraction",
            "stage": "extraction",
            "data": {"events": ["查天气"]},
        }
        yield {
            "type": "final",
            "stage": "done",
            "data": {
                "final_response": "北京明天晴，适合出行。",
                "trace_id": "trace-sdk-integration",
                "span_id": "span-sdk-integration",
            },
        }

    mock_orch.iter_request_stream = fake_stream

    port = _free_port()
    config = uvicorn.Config(
        api_mod.app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)

    with patch.object(api_mod, "_get_orchestrator", AsyncMock(return_value=mock_orch)):
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_health(base_url)
        try:
            yield base_url
        finally:
            server.should_exit = True
            thread.join(timeout=5.0)


@pytest.mark.skipif(_npm_executable() is None, reason="npm not installed")
def test_router_client_sdk_integration(mock_router_api_server):
    subprocess.run("npm install", cwd=PKG_ROOT, check=True, capture_output=True, shell=True)
    subprocess.run("npm run build", cwd=PKG_ROOT, check=True, capture_output=True, shell=True)

    env = os.environ.copy()
    env["ROUTER_CLIENT_BASE_URL"] = mock_router_api_server
    result = subprocess.run(
        "node --test tests/integration.test.mjs",
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "router-client integration failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
