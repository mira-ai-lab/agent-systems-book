#!/usr/bin/env python3
"""router-client SDK 联调冒烟（Phase 26.1）。

对已在运行的 platform API 执行 Node integration 测试::

    python scripts/run_api.py   # 终端 1
    python scripts/smoke_router_client.py

可选环境变量::

    ROUTER_CLIENT_BASE_URL   默认 http://127.0.0.1:8780
    ROUTER_CLIENT_API_KEY    对应服务端 API_KEYS
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = ROOT / "packages" / "router-client"


def _npm() -> str:
    return shutil.which("npm") or shutil.which("npm.cmd") or "npm"


def _wait_health(base_url: str, timeout_sec: float = 15.0) -> None:
    import time

    url = f"{base_url.rstrip('/')}/health"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    raise SystemExit(f"API not reachable: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test @agent-platform/router-client")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ROUTER_CLIENT_BASE_URL", "http://127.0.0.1:8780"),
        help="Platform API base URL",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    if not PKG_ROOT.is_dir():
        print(f"missing package: {PKG_ROOT}", file=sys.stderr)
        return 1

    print(f"Checking API health at {base_url} ...")
    _wait_health(base_url)

    npm = _npm()
    subprocess.run(f'"{npm}" install', cwd=PKG_ROOT, check=True, shell=True)
    subprocess.run(f'"{npm}" run build', cwd=PKG_ROOT, check=True, shell=True)

    env = os.environ.copy()
    env["ROUTER_CLIENT_BASE_URL"] = base_url
    print("Running router-client integration tests ...")
    subprocess.run(
        "node --test tests/integration.test.mjs",
        cwd=PKG_ROOT,
        env=env,
        check=True,
        shell=True,
    )
    print("router-client smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
