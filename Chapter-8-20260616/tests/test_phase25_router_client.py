"""Phase 25 P3：前端 SDK @agent-platform/router-client。"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "packages" / "router-client"


def test_router_client_package_layout():
    assert (PKG_ROOT / "package.json").is_file()
    assert (PKG_ROOT / "src" / "index.ts").is_file()
    assert (PKG_ROOT / "src" / "client.ts").is_file()
    assert (PKG_ROOT / "src" / "sse.ts").is_file()
    assert (PKG_ROOT / "README.md").is_file()


def test_router_client_package_metadata():
    payload = json.loads((PKG_ROOT / "package.json").read_text(encoding="utf-8"))
    assert payload["name"] == "@agent-platform/router-client"
    assert payload["version"]
    assert "build" in payload["scripts"]
    assert "test" in payload["scripts"]


def test_router_client_exports_route_and_stream():
    client_src = (PKG_ROOT / "src" / "client.ts").read_text(encoding="utf-8")
    index_src = (PKG_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
    assert "async route(" in client_src
    assert "routeStream" in client_src
    assert "submitJob" in client_src
    assert "getJob" in client_src
    assert "/v1/chat/stream" in client_src
    assert "/v1/jobs" in client_src
    assert "createRouterClient" in index_src


def _npm_executable() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


@pytest.mark.skipif(_npm_executable() is None, reason="npm not installed")
def test_router_client_build_and_unit_tests():
    subprocess.run("npm install", cwd=PKG_ROOT, check=True, capture_output=True, shell=True)
    subprocess.run("npm test", cwd=PKG_ROOT, check=True, capture_output=True, shell=True)
    assert (PKG_ROOT / "dist" / "index.js").is_file()
    assert (PKG_ROOT / "dist" / "index.d.ts").is_file()
