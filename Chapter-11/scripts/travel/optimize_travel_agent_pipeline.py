#!/usr/bin/env python3
"""Travel 子 Agent mini-pipeline 串联优化（Agent-B3）。

固定 fixture subtask 依次调用 Weather → Hotel → Flight 等，
按 slot 顺序优化 system_prompt，rollback 以 pipeline dev 分为准。

用法::

    # 默认 slots：WeatherAgent,HotelAgent,FlightAgent
    python scripts/optimize_travel_agent_pipeline.py --max-steps 3

    # 自定义串联顺序
    python scripts/optimize_travel_agent_pipeline.py --slots WeatherAgent,HotelAgent,RestaurantAgent

    pip install -e ".[evolution]"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.optimization.agent_mini_pipeline import (
    parse_mini_pipeline_slots,
    run_mini_pipeline_optimization,
)
from agent_framework.optimization.agent_prompt_store import optimized_agent_prompts_path
from agent_framework.optimization.agents.mini_pipeline.fixtures import load_mini_pipeline_cases
from agent_framework.optimization.core.save import save_multi_agent_optimization_artifacts
from agent_framework.optimization.optimizers.textgrad_agent.optimize import default_agent_report_path


def _resolve_model(env_name: str, fallback_env: str, default: str) -> str:
    return (
        os.getenv(env_name, "").strip()
        or os.getenv(fallback_env, "").strip()
        or default
    )


def _print_slot_result(agent_name: str, result) -> None:
    print(
        f"  [{agent_name}] baseline_pipeline_dev={result.baseline_dev_score:.3f} "
        f"best_pipeline_dev={result.best_dev_score:.3f} optimizer={result.optimizer}"
    )
    for step in result.steps:
        flag = "ACCEPT" if step.accepted else "REJECT"
        print(
            f"    step={step.step} {flag} train_pipeline={step.train_average:.3f} "
            f"candidate_pipeline_dev={step.candidate_dev_average:.3f} "
            f"step_failures={step.failure_count}"
        )


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    slots = parse_mini_pipeline_slots(args.slots)
    fixtures = load_mini_pipeline_cases(args.fixtures)
    executor_model = _resolve_model("EXECUTOR_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    optimizer_model = _resolve_model("OPTIMIZER_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    output_path = args.output or optimized_agent_prompts_path(fixtures.locale)
    report_dir = output_path.parent

    print(
        f"start mini-pipeline optimization: slots={slots} backend=textgrad_agent_mini_pipeline "
        f"train={args.train_split} dev={args.dev_split} max_steps={args.max_steps} "
        f"rollback={not args.no_rollback}"
    )
    print(f"executor_model={executor_model} optimizer_model={optimizer_model}")

    pipeline_output = await run_mini_pipeline_optimization(
        slots=slots,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=args.max_steps,
        failure_threshold=args.failure_threshold,
        step_failure_threshold=args.step_failure_threshold,
        rollback=not args.no_rollback,
        train_split=args.train_split,
        dev_split=args.dev_split,
    )

    save_multi_agent_optimization_artifacts(
        pipeline_output.results,
        locale=fixtures.locale,
        output_path=output_path,
        report_dir=report_dir,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        extra_metadata={
            "backend": "textgrad_agent_mini_pipeline",
            "slots": slots,
        },
    )

    for slot in slots:
        _print_slot_result(slot, pipeline_output.results[slot])

    print(f"saved agent prompts: {output_path}")
    for slot in slots:
        print(f"saved report: {report_dir / Path(default_agent_report_path(slot, fixtures.locale)).name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel mini-pipeline 子 Agent 串联优化（Agent-B3）")
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument(
        "--slots",
        default="default",
        help="串联 slot：default / all / 逗号分隔 Agent 名",
    )
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "all"])
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "all"])
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--step-failure-threshold", type=float, default=0.8)
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--output", type=Path)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
