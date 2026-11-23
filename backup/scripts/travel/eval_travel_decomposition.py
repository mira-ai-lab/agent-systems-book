#!/usr/bin/env python3
"""Travel 任务拆解 baseline 评测（fixtures + 规则打分 + LLM 路由覆盖）。

拆解质量仍用规则打分；Agent 覆盖项改为对 live 拆解结果跑 ``route_to_agents``，
用 LLM 路由结论与 ``expect.mappable_agents`` 比对（不再使用 guess_rules 启发式）。

用法::

    python scripts/eval_travel_decomposition.py
    python scripts/eval_travel_decomposition.py --split dev
    python scripts/eval_travel_decomposition.py --split all --json
    python scripts/eval_travel_decomposition.py --output reports/decomp_dev.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.domain.task_planner import TaskPlanner
from agent_framework.optimization.prompt_store import load_optimized_prompt_payload
from agent_framework.optimization.decomposition.evaluator import evaluate_decomposition_benchmark
from agent_framework.optimization.decomposition.fixtures import (
    default_fixtures_path,
    load_decomposition_fixtures,
)
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def _print_report(report) -> None:
    print(
        f"version={getattr(report, 'version', 'n/a')} domain={report.domain} "
        f"locale={report.locale} split={report.split} "
        f"cases={report.case_count} avg_score={report.average_score:.3f}"
    )
    for item in report.cases:
        status = "PASS" if item.score.total >= 0.8 else "FAIL"
        print(
            f"[{status}] {item.case_id} score={item.score.total:.3f} "
            f"subtasks={item.score.subtask_count}"
        )
        if item.score.details:
            print(f"  details: {'; '.join(item.score.details)}")
        if item.sub_steps:
            preview = " | ".join(item.sub_steps[:3])
            print(f"  steps: {preview}")
        if item.routed_agents:
            print(f"  routed: {', '.join(item.routed_agents)}")
        if item.execution_order:
            print(f"  order: {' -> '.join(item.execution_order)}")
        if item.depends_map:
            dep_preview = "; ".join(
                f"{task_id}->{','.join(deps) or '-'}"
                for task_id, deps in item.depends_map.items()
                if deps
            )
            if dep_preview:
                print(f"  deps: {dep_preview}")


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    fixtures = load_decomposition_fixtures(args.fixtures)
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build(locale=fixtures.locale, use_optimized=False)

    if args.prompt_file:
        prompt_payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        custom_prompt = str(prompt_payload.get("decomposition_prompt") or "").strip()
        if not custom_prompt:
            raise ValueError(f"{args.prompt_file} 缺少 decomposition_prompt")
        prompts = replace(prompts, decomposition_prompt=custom_prompt)
    elif args.use_optimized:
        payload = load_optimized_prompt_payload(fixtures.locale)
        overrides = {
            key: str(payload[key])
            for key in ("decomposition_prompt", "agent_routing")
            if str(payload.get(key) or "").strip()
        }
        if overrides:
            prompts = replace(prompts, **overrides)

    planner = TaskPlanner(create_llm(temperature=0), registry, prompts)
    report = await evaluate_decomposition_benchmark(
        planner,
        registry=registry,
        fixtures=fixtures,
        split=args.split,
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
    parser = argparse.ArgumentParser(description="Travel 任务拆解 baseline 评测")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=default_fixtures_path(),
        help="benchmark fixture JSON 路径",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "test", "all"],
        help="评测 split",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="可选：覆盖 decomposition_prompt 的 JSON 文件",
    )
    parser.add_argument(
        "--use-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="加载 data/benchmark/travel_planner/optimized/{locale}.json",
    )
    parser.add_argument("--json", action="store_true", help="JSON 报告输出")
    parser.add_argument("--output", type=Path, help="报告写入文件")
    parser.add_argument(
        "--min-avg",
        type=float,
        default=0.0,
        help="平均分低于该阈值时返回非 0 退出码（CI 可选）",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
