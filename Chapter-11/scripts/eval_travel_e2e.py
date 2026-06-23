#!/usr/bin/env python3
"""Travel 端到端 baseline 评测（完整编排：Router/LangGraph → 子 Agent → final_response）。

用法::

    python scripts/eval_travel_e2e.py --split dev
    python scripts/eval_travel_e2e.py --split dev --profile legacy
    python scripts/eval_travel_e2e.py --split test --json --output reports/e2e_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.optimization.decomposition.fixtures import (
    default_fixtures_path,
    load_decomposition_fixtures,
)
from agent_framework.optimization.e2e.evaluator import evaluate_e2e_benchmark
from agent_framework.optimization.e2e.runtime import build_e2e_orchestrator
from agent_framework.optimization.prompt_store import load_optimized_prompt_payload


def _print_report(report) -> None:
    print(
        f"domain={report.domain} locale={report.locale} split={report.split} "
        f"profile={report.profile} cases={report.case_count} avg_score={report.average_score:.3f}"
    )
    for item in report.cases:
        status = "PASS" if item.score.total >= 0.8 else "FAIL"
        agents = ", ".join(item.invoked_agents) or "(none)"
        print(
            f"[{status}] {item.case_id} score={item.score.total:.3f} "
            f"completed={item.completed_subtasks} agents={agents}"
        )
        if item.score.details:
            print(f"  details: {'; '.join(item.score.details)}")
        if item.final_response:
            preview = item.final_response.replace("\n", " ")[:160]
            print(f"  response: {preview}")


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    fixtures = load_decomposition_fixtures(args.fixtures)
    overrides = {}
    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        for key in ("decomposition_prompt", "agent_routing"):
            if str(payload.get(key) or "").strip():
                overrides[key] = str(payload[key])
    elif args.use_optimized:
        payload = load_optimized_prompt_payload(fixtures.locale)
        overrides = {
            key: str(payload[key])
            for key in ("decomposition_prompt", "agent_routing")
            if str(payload.get(key) or "").strip()
        }

    orchestrator = build_e2e_orchestrator(
        create_llm(temperature=0),
        locale=fixtures.locale,
        profile=args.profile,
        enable_memory=not args.no_memory,
        enable_guess_agent=not args.no_guess_agent,
        prompt_overrides=overrides or None,
        use_optimized=args.use_optimized and not args.prompt_file,
    )

    print(
        f"start e2e eval: split={args.split} profile={args.profile} "
        f"use_optimized={args.use_optimized and not args.prompt_file}"
    )
    report = await evaluate_e2e_benchmark(
        orchestrator,
        fixtures=fixtures,
        split=args.split,
        profile=args.profile,
        timeout_sec=args.timeout,
    )

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
    parser = argparse.ArgumentParser(description="Travel 端到端 baseline 评测")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument(
        "--profile",
        default="workflow",
        choices=["workflow", "legacy"],
        help="workflow=Router+FixedGraph；legacy=LangGraph 内完整 TaskPlanner",
    )
    parser.add_argument("--prompt-file", type=Path, help="覆盖 optimized prompt 的 JSON")
    parser.add_argument(
        "--use-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="加载 data/benchmark/travel_planner/optimized/{locale}.json",
    )
    parser.add_argument("--no-memory", action="store_true", help="禁用长期记忆（推荐评测时开启）")
    parser.add_argument("--no-guess-agent", action="store_true", help="禁用 guess_agent 路由兜底")
    parser.add_argument("--timeout", type=float, help="单 case 超时秒数")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-avg", type=float, default=0.0)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
