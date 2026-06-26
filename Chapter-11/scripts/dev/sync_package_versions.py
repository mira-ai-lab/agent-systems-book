#!/usr/bin/env python3
"""Platform semver 同步与校验（Phase 26.4）。

``agent-platform``（Python）与 ``packages/*``（npm）共用同一 semver。

用法::

    python scripts/sync_package_versions.py --check
    python scripts/sync_package_versions.py --check --json
    python scripts/sync_package_versions.py --sync          # 以 pyproject.toml 为准对齐 npm 包
    python scripts/sync_package_versions.py --sync 0.22.0   # 全部 bump 到指定版本
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
_PYPROJECT_VERSION_LINE = re.compile(r'^version\s*=\s*"[^"]+"', re.MULTILINE)


@dataclass(frozen=True)
class VersionTarget:
    name: str
    path: Path
    file_kind: str  # "toml" | "json"


def version_targets(root: Path = ROOT) -> list[VersionTarget]:
    return [
        VersionTarget("agent-platform", root / "pyproject.toml", "toml"),
        VersionTarget(
            "@agent-platform/router-client",
            root / "packages/router-client/package.json",
            "json",
        ),
        VersionTarget(
            "@agent-platform/demo-web",
            root / "packages/demo-web/package.json",
            "json",
        ),
    ]


def read_version(target: VersionTarget) -> str:
    text = target.path.read_text(encoding="utf-8")
    if target.file_kind == "toml":
        match = _PYPROJECT_VERSION.search(text)
        if not match:
            raise ValueError(f"version not found in {target.path}")
        return match.group(1)
    payload = json.loads(text)
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"version missing in {target.path}")
    return version


def write_version(target: VersionTarget, version: str) -> None:
    if target.file_kind == "toml":
        text = target.path.read_text(encoding="utf-8")
        new_text, count = _PYPROJECT_VERSION_LINE.subn(
            f'version = "{version}"',
            text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"failed to update version in {target.path}")
        target.path.write_text(new_text, encoding="utf-8")
        return

    payload = json.loads(target.path.read_text(encoding="utf-8"))
    payload["version"] = version
    target.path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_versions(root: Path = ROOT) -> dict[str, str]:
    return {target.name: read_version(target) for target in version_targets(root)}


def check_alignment(root: Path = ROOT) -> tuple[bool, dict[str, str]]:
    versions = collect_versions(root)
    unique = set(versions.values())
    return len(unique) == 1, versions


def sync_versions(version: str | None = None, root: Path = ROOT) -> str:
    targets = version_targets(root)
    canonical = version or read_version(targets[0])
    for target in targets:
        write_version(target, canonical)
    return canonical


def main() -> int:
    parser = argparse.ArgumentParser(description="Platform semver sync (Phase 26.4)")
    parser.add_argument("--check", action="store_true", help="Exit 1 if versions differ")
    parser.add_argument("--sync", nargs="?", const="", metavar="VERSION", help="Align all targets")
    parser.add_argument("--json", action="store_true", help="JSON output for --check")
    args = parser.parse_args()

    if args.sync is not None:
        new_version = args.sync or None
        applied = sync_versions(new_version)
        print(f"synced platform semver to {applied}")
        for name, value in collect_versions().items():
            print(f"  {name}: {value}")
        return 0

    ok, versions = check_alignment()
    if args.json:
        print(
            json.dumps(
                {
                    "aligned": ok,
                    "versions": versions,
                    "canonical": next(iter(versions.values())) if ok else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if ok:
            version = next(iter(versions.values()))
            print(f"platform semver aligned: {version}")
            for name, value in versions.items():
                print(f"  {name}: {value}")
        else:
            print("platform semver mismatch:")
            for name, value in versions.items():
                print(f"  {name}: {value}")
            print("fix: python scripts/sync_package_versions.py --sync")

    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
