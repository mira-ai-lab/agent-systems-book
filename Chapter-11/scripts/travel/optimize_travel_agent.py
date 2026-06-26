#!/usr/bin/env python3
"""Travel 子 Agent system_prompt 优化（Agent-B1 单 Agent，Agent-B2 五 Agent 并列）。

用法::

    # B1：仅 FlightAgent
    python scripts/optimize_travel_agent.py --agent FlightAgent --max-steps 3

    # B2：5 个子 Agent 并列优化（默认 parallel）
    python scripts/optimize_travel_agent.py --agent all --max-steps 3

    # B2：顺序优化（降低 LLM 并发）
    python scripts/optimize_travel_agent.py --agent all --sequential --max-steps 3

    # 打印每条 case 评测进度
    python scripts/optimize_travel_agent.py --agent FlightAgent --verbose --max-steps 1

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
from agent_framework.optimization.agent_pipeline import parse_agent_slots, run_agent_optimization
from agent_framework.optimization.agent_prompt_store import optimized_agent_prompts_path
from agent_framework.optimization.agents.evaluator import make_case_eval_progress_printer
from agent_framework.optimization.agents.fixtures import load_single_agent_cases
from agent_framework.optimization.agents.runtime import resolve_optimization_start_template
from agent_framework.optimization.core.save import (
    save_agent_optimization_artifacts,
    save_multi_agent_optimization_artifacts,
)
from agent_framework.optimization.optimizers.textgrad_agent.flight import (
    FLIGHT_AGENT_NAME,
    default_flight_agent_report_path,
    optimize_flight_agent_prompt_graph,
)
from agent_framework.optimization.optimizers.textgrad_agent.optimize import default_agent_report_path


def _resolve_model(env_name: str, fallback_env: str, default: str) -> str:
    return (
        os.getenv(env_name, "").strip()
        or os.getenv(fallback_env, "").strip()
        or default
    )


def _print_result(agent_name: str, result) -> None:
    print(
        f"  [{agent_name}] baseline_dev={result.baseline_dev_score:.3f} "
        f"best_dev={result.best_dev_score:.3f} optimizer={result.optimizer}"
    )
    for step in result.steps:
        flag = "ACCEPT" if step.accepted else "REJECT"
        print(
            f"    step={step.step} {flag} train={step.train_average:.3f} "
            f"candidate_dev={step.candidate_dev_average:.3f} failures={step.failure_count}"
        )


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    agents = parse_agent_slots(args.agent)
    fixtures = load_single_agent_cases(args.fixtures)
    executor_model = _resolve_model("EXECUTOR_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    optimizer_model = _resolve_model("OPTIMIZER_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    parallel = not args.sequential
    output_path = args.output or optimized_agent_prompts_path(fixtures.locale)
    report_dir = output_path.parent

    print(
        f"start agent optimization: agents={agents} backend=textgrad_agent_graph "
        f"parallel={parallel} train={args.train_split} dev={args.dev_split} "
        f"max_steps={args.max_steps} rollback={not args.no_rollback}"
    )
    print(f"executor_model={executor_model} optimizer_model={optimizer_model}")
    if args.start_from_locales:
        print("start_prompt=locales (ignoring optimized override for optimization loop)")
    else:
        print("start_prompt=optimized_or_locales (same source as eval_travel_agents)")
    if args.verbose:
        print("verbose: printing one line after each case evaluation", flush=True)

    optimize_kwargs = dict(
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=args.max_steps,
        failure_threshold=args.failure_threshold,
        rollback=not args.no_rollback,
        train_split=args.train_split,
        dev_split=args.dev_split,
    )

    if len(agents) == 1:
        # 单 Agent：沿用 B1 路径，报告写到默认单文件
        agent_name = agents[0]
        start_template = resolve_optimization_start_template(
            agent_name,
            locale=fixtures.locale,
            start_from_locales=args.start_from_locales,
        )
        if args.verbose:
            optimize_kwargs["on_case_evaluated"] = make_case_eval_progress_printer(agent_name)

        if agent_name == FLIGHT_AGENT_NAME:
            result = await optimize_flight_agent_prompt_graph(
                system_prompt_template=start_template,
                **optimize_kwargs,
            )
            report_path = Path(args.report_output or default_flight_agent_report_path(fixtures.locale))
        else:
            from agent_framework.optimization.optimizers.textgrad_agent.optimize import (
                optimize_agent_prompt_graph,
            )

            result = await optimize_agent_prompt_graph(
                agent_name=agent_name,
                system_prompt_template=start_template,
                **optimize_kwargs,
            )
            report_path = Path(
                args.report_output or default_agent_report_path(agent_name, fixtures.locale)
            )

        save_agent_optimization_artifacts(
            result,
            agent_name=agent_name,
            locale=fixtures.locale,
            output_path=output_path,
            report_path=report_path,
            executor_model=executor_model,
            optimizer_model=optimizer_model,
            extra_metadata={"backend": "textgrad_agent_graph"},
        )
        _print_result(agent_name, result)
        print(f"saved agent prompts: {output_path}")
        print(f"saved report: {report_path}")
        return 0

    # 多 Agent：B2 pipeline
    pipeline_output = await run_agent_optimization(
        agents=agents,
        parallel=parallel,
        verbose=args.verbose,
        start_from_locales=args.start_from_locales,
        **{k: v for k, v in optimize_kwargs.items() if k != "on_case_evaluated"},
    )

    save_multi_agent_optimization_artifacts(
        pipeline_output.results,
        locale=fixtures.locale,
        output_path=output_path,
        report_dir=report_dir,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        extra_metadata={"backend": "textgrad_agent_graph", "parallel": parallel},
    )

    for agent_name in agents:
        _print_result(agent_name, pipeline_output.results[agent_name])

    print(f"saved agent prompts: {output_path}")
    print(f"saved reports under: {report_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel 子 Agent prompt 优化（Agent-B1/B2）")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="benchmark fixtures 路径",
    )
    parser.add_argument(
        "--agent",
        default=FLIGHT_AGENT_NAME,
        help="Agent 名，或 all，或逗号分隔列表（如 FlightAgent,WeatherAgent）",
    )
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "all"])
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "all"])
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="顺序优化各 Agent（默认并列 parallel）",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path, help="仅单 Agent 模式有效")
    parser.add_argument(
        "--start-from-locales",
        action="store_true",
        help="优化循环从 locales 基线起跑（忽略 optimized 破坏版）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="每评完一条 benchmark case 打印一行进度（含 TextGrad forward）",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
