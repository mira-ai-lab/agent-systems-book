"""Phase 26：demo-web 布局 + CI 产物检查。"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = ROOT / "packages" / "demo-web"
ROUTER_ROOT = ROOT / "packages" / "router-client"


def test_demo_web_package_layout():
    assert (DEMO_ROOT / "package.json").is_file()
    assert (DEMO_ROOT / "vite.config.ts").is_file()
    assert (DEMO_ROOT / "index.html").is_file()
    assert (DEMO_ROOT / "src" / "main.ts").is_file()
    assert (DEMO_ROOT / "README.md").is_file()


def test_demo_web_depends_on_router_client():
    payload = json.loads((DEMO_ROOT / "package.json").read_text(encoding="utf-8"))
    dep = payload.get("dependencies", {}).get("@agent-platform/router-client", "")
    assert dep == "file:../router-client"
    main_src = (DEMO_ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    assert "createRouterClient" in main_src
    assert "routeStream" in main_src


def _npm_executable() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


@pytest.mark.skipif(_npm_executable() is None, reason="npm not installed")
def test_demo_web_build():
    subprocess.run("npm ci", cwd=ROUTER_ROOT, check=True, capture_output=True, shell=True)
    subprocess.run("npm run build", cwd=ROUTER_ROOT, check=True, capture_output=True, shell=True)
    subprocess.run("npm install", cwd=DEMO_ROOT, check=True, capture_output=True, shell=True)
    subprocess.run("npm run build", cwd=DEMO_ROOT, check=True, capture_output=True, shell=True)
    assert (DEMO_ROOT / "dist" / "index.html").is_file()
