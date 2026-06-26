#!/usr/bin/env python3
"""Travel 子任务路由 baseline 评测。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.optimization.decomposition.fixtures import default_fixtures_path, load_decomposition_fixtures
from agent_framework.optimization.planner_runtime import build_planner
from agent_framework.optimization.prompt_store import load_optimized_prompt_payload
from agent_framework.optimization.routing.evaluator import evaluate_routing_benchmark
from domains.travel.specs import create_travel_registry_stub


def _print_report(report) -> None:
    print(
        f"domain={report.domain} locale={report.locale} split={report.split} "
        f"cases={report.case_count} avg_score={report.average_score:.3f}"
    )
    for item in report.cases:
        status = "PASS" if item.score.total >= 0.8 else "FAIL"
        agents = ", ".join(f"{t.get('task_id')}->{t.get('agent')}" for t in item.subtasks[:4])
        print(f"[{status}] {item.case_id} score={item.score.total:.3f} routing={agents}")
        if item.score.details:
            print(f"  details: {'; '.join(item.score.details)}")


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    fixtures = load_decomposition_fixtures(args.fixtures)
    registry = create_travel_registry_stub()
    overrides = {}
    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        if payload.get("agent_routing"):
            overrides["agent_routing"] = payload["agent_routing"]
        if payload.get("decomposition_prompt"):
            overrides["decomposition_prompt"] = payload["decomposition_prompt"]
    elif args.use_optimized:
        payload = load_optimized_prompt_payload(fixtures.locale)
        overrides = {
            key: str(payload[key])
            for key in ("decomposition_prompt", "agent_routing")
            if str(payload.get(key) or "").strip()
        }

    planner = build_planner(
        create_llm(temperature=0),
        registry,
        locale=fixtures.locale,
        prompt_overrides=overrides or None,
        use_optimized=args.use_optimized and not args.prompt_file,
    )
    report = await evaluate_routing_benchmark(planner, fixtures=fixtures, split=args.split)

    if args.json:
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(f"report saved to {args.output}")
        else:
            print(payload)
    else:
        _print_report(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"report saved to {args.output}")

    return 0 if report.average_score >= args.min_avg else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel agent_routing baseline 评测")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--use-optimized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-avg", type=float, default=0.0)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
