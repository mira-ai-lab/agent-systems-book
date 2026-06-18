#!/usr/bin/env python3
"""六维度产品化就绪度自检（Phase 24.20）。

用法::

    python scripts/product_readiness_check.py
    python scripts/product_readiness_check.py --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class DimensionScore:
    name: str
    label: str
    checks: list[tuple[str, bool]]
    target_pct: int

    @property
    def score_pct(self) -> int:
        if not self.checks:
            return 0
        passed = sum(1 for _, ok in self.checks if ok)
        return round(100 * passed / len(self.checks))

    def meets_target(self) -> bool:
        return self.score_pct >= self.target_pct


def _exists(relative: str) -> bool:
    return (ROOT / relative).is_file()


def _module_has_attr(module_path: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return False
    return hasattr(mod, attr)


def evaluate_dimensions() -> list[DimensionScore]:
    return [
        DimensionScore(
            name="default_mindset",
            label="1 默认心智",
            target_pct=95,
            checks=[
                ("bootstrap.route() 入口", _module_has_attr("agent_framework.bootstrap", "route")),
                ("entry.py route 实现", _exists("agent_framework/bootstrap/entry.py")),
                ("run_demo 默认 customer_service", "customer_service" in (ROOT / "scripts/run_demo.py").read_text(encoding="utf-8")),
                ("platform_domain_router 跨域推断", _exists("agent_framework/router/platform_domain_router.py")),
            ],
        ),
        DimensionScore(
            name="router_engine",
            label="2 核心能力",
            target_pct=95,
            checks=[
                ("RouterEngine", _module_has_attr("agent_framework.router.engine", "RouterEngine")),
                ("knowledge_routing 阶段", _exists("agent_framework/router/stages/knowledge_routing.py")),
                ("Embedding 后端抽象", _exists("agent_framework/router/kb/backends/factory.py")),
                ("Chroma KB + ingest CLI", _exists("scripts/ingest_knowledge.py")),
                ("KB recall benchmark", _exists("scripts/benchmark_knowledge_recall.py")),
                ("KB 管理 API scoring", _exists("agent_framework/router/kb/scoring.py")),
                ("routing observability", _exists("agent_framework/router/observability.py")),
                ("router-client SDK", (ROOT / "packages/router-client/package.json").is_file()),
                ("demo-web UI", (ROOT / "packages/demo-web/package.json").is_file()),
                ("sdk_integration.md", _exists("docs/sdk_integration.md")),
                ("semver sync script", _exists("scripts/sync_package_versions.py")),
            ],
        ),
        DimensionScore(
            name="orchestration",
            label="3 编排",
            target_pct=98,
            checks=[
                ("profile auto/workflow/adaptive/hybrid", _exists("agent_framework/router/profile.py")),
                ("RouterOrchestrator", _exists("agent_framework/orchestration/router_orchestrator.py")),
                ("profile_reason metadata", "profile_reason" in (ROOT / "agent_framework/router/engine.py").read_text(encoding="utf-8")),
                ("Supervisor handoff 流式", "handoff_event" in (ROOT / "agent_framework/stream/events.py").read_text(encoding="utf-8")),
            ],
        ),
        DimensionScore(
            name="registry",
            label="4 扩展",
            target_pct=95,
            checks=[
                ("dynamic_registry", _exists("agent_framework/domain/dynamic_registry.py")),
                ("agent_catalog", _exists("agent_framework/domain/agent_catalog.py")),
                ("alias_of 跨域引用", "alias_of" in (ROOT / "agent_framework/domain/dynamic_registry.py").read_text(encoding="utf-8")),
                ("registry.updated 事件", _exists("agent_framework/domain/registry_events.py")),
                ("registry_federation", _exists("agent_framework/domain/registry_federation.py")),
            ],
        ),
        DimensionScore(
            name="travel_positioning",
            label="5 travel 定位",
            target_pct=95,
            checks=[
                ("docs/domains.md", _exists("docs/domains.md")),
                ("travel 能力展示叙事", "能力展示" in (ROOT / "docs/domains.md").read_text(encoding="utf-8")),
                ("run_demo travel 标注", "--domain travel" in (ROOT / "scripts/run_demo.py").read_text(encoding="utf-8")),
                ("travel 插件存在", _exists("domains/travel/plugin.py")),
            ],
        ),
        DimensionScore(
            name="i18n",
            label="6 i18n",
            target_pct=92,
            checks=[
                ("domain locales JSON", _exists("domains/customer_service/locales/en.json")),
                ("agent locales JSON", _exists("domains/travel/agents/locales/en.json")),
                ("locale_loader agent_system_prompt", _module_has_attr("agent_framework.domain.locale_loader", "agent_system_prompt")),
                ("test_phase24_i18n", _exists("tests/test_phase24_i18n.py")),
            ],
        ),
    ]


def overall_score(dimensions: list[DimensionScore]) -> int:
    return round(sum(d.score_pct for d in dimensions) / len(dimensions))


def main() -> int:
    parser = argparse.ArgumentParser(description="六维度产品化就绪度自检")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    dimensions = evaluate_dimensions()
    overall = overall_score(dimensions)

    if args.json:
        payload = {
            "overall_pct": overall,
            "dimensions": [
                {
                    "name": d.name,
                    "label": d.label,
                    "score_pct": d.score_pct,
                    "target_pct": d.target_pct,
                    "meets_target": d.meets_target(),
                    "checks": [{"name": n, "passed": ok} for n, ok in d.checks],
                }
                for d in dimensions
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if all(d.meets_target() for d in dimensions) else 1

    print("Phase 24 六维度产品化就绪度")
    print("=" * 48)
    for d in dimensions:
        status = "OK" if d.meets_target() else "!!"
        print(f"[{status}] {d.label}: {d.score_pct}% (目标 {d.target_pct}%)")
        for name, ok in d.checks:
            mark = "PASS" if ok else "FAIL"
            print(f"      [{mark}] {name}")
    print("-" * 48)
    print(f"综合: {overall}%")
    return 0 if all(d.meets_target() for d in dimensions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
