"""Phase 26.4：platform semver 与 npm 包版本同步。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_sync_module():
    path = ROOT / "scripts" / "dev" / "sync_package_versions.py"
    mod_name = "sync_package_versions"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_platform_versions_are_aligned():
    mod = _load_sync_module()
    ok, versions = mod.check_alignment()
    assert ok, versions
    assert versions["agent-platform"] == versions["@agent-platform/router-client"]
    assert versions["@agent-platform/router-client"] == versions["@agent-platform/demo-web"]


def test_sync_versions_updates_all_targets(tmp_path):
    mod = _load_sync_module()

    (tmp_path / "pyproject.toml").write_text(
        'name = "agent-platform"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    router_dir = tmp_path / "packages" / "router-client"
    demo_dir = tmp_path / "packages" / "demo-web"
    router_dir.mkdir(parents=True)
    demo_dir.mkdir(parents=True)
    router_dir.joinpath("package.json").write_text(
        json.dumps({"name": "@agent-platform/router-client", "version": "0.9.0"}),
        encoding="utf-8",
    )
    demo_dir.joinpath("package.json").write_text(
        json.dumps({"name": "@agent-platform/demo-web", "version": "0.8.0"}),
        encoding="utf-8",
    )

    applied = mod.sync_versions("2.3.4", root=tmp_path)
    assert applied == "2.3.4"

    ok, versions = mod.check_alignment(root=tmp_path)
    assert ok
    assert versions["agent-platform"] == "2.3.4"


def test_sync_without_arg_uses_pyproject_version(tmp_path):
    mod = _load_sync_module()

    (tmp_path / "pyproject.toml").write_text('version = "0.21.0"\n', encoding="utf-8")
    router_dir = tmp_path / "packages" / "router-client"
    demo_dir = tmp_path / "packages" / "demo-web"
    router_dir.mkdir(parents=True)
    demo_dir.mkdir(parents=True)
    router_dir.joinpath("package.json").write_text(
        json.dumps({"name": "@agent-platform/router-client", "version": "0.20.0"}),
        encoding="utf-8",
    )
    demo_dir.joinpath("package.json").write_text(
        json.dumps({"name": "@agent-platform/demo-web", "version": "0.19.0"}),
        encoding="utf-8",
    )

    applied = mod.sync_versions(root=tmp_path)
    assert applied == "0.21.0"
    ok, versions = mod.check_alignment(root=tmp_path)
    assert ok
