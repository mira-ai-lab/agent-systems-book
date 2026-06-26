#!/usr/bin/env python3
"""一键优化 Travel Planner prompts（decomposition → routing）。

用法::

    # local optimizer（默认）
    python scripts/optimize_travel_planner.py --backend local --max-steps 5

    # textgrad 库版（需 pip install -e ".[evolution]"）
    python scripts/optimize_travel_planner.py --backend textgrad_lib --max-steps 3

    # textgrad 计算图版（TaskPlanner 三步接 StringBasedFunction，Phase B1）
    python scripts/optimize_travel_planner.py --backend textgrad_graph --max-steps 3

    # E2E graph loss 驱动（Phase B2，需 textgrad_graph + --objective e2e）
    python scripts/optimize_travel_planner.py --backend textgrad_graph --objective e2e --max-steps 1 --train-split dev --dev-split dev

    # 只优化 routing（使用当前 decomposition_prompt）
    python scripts/optimize_travel_planner.py --slots routing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.optimization.core.save import save_planner_optimization_artifacts
from agent_framework.optimization.decomposition.fixtures import default_fixtures_path, load_decomposition_fixtures
from agent_framework.optimization.objective import parse_optimization_objective
from agent_framework.optimization.planner_pipeline import parse_planner_slots, run_planner_optimization
from agent_framework.optimization.prompt_store import optimized_prompts_path
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def _resolve_model(env_name: str, fallback_env: str, default: str) -> str:
    return (
        os.getenv(env_name, "").strip()
        or os.getenv(fallback_env, "").strip()
        or default
    )


def _print_slot_result(label: str, result) -> None:
    if result is None:
        return
    print(
        f"{label}: baseline_dev={result.baseline_dev_score:.3f} "
        f"best_dev={result.best_dev_score:.3f} optimizer={result.optimizer} steps={len(result.steps)}"
    )
    for step in result.steps:
        flag = "ACCEPT" if step.accepted else "REJECT"
        print(
            f"  [{label}] step={step.step} {flag} train={step.train_average:.3f} "
            f"candidate_dev={step.candidate_dev_average:.3f} failures={step.failure_count}"
        )


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    fixtures = load_decomposition_fixtures(args.fixtures)
    registry = create_travel_registry_stub()
    base_prompts = TravelPrompts.build(locale=fixtures.locale, use_optimized=False)
    decomposition_prompt = base_prompts.decomposition_prompt
    agent_routing = base_prompts.agent_routing
    slots = parse_planner_slots(args.slots)
    objective = parse_optimization_objective(args.objective)

    if objective == "e2e" and args.backend != "textgrad_graph":
        raise SystemExit("objective=e2e 仅支持 --backend textgrad_graph（Phase B2 E2E graph）")

    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        if payload.get("decomposition_prompt"):
            decomposition_prompt = str(payload["decomposition_prompt"]).strip()
        if payload.get("agent_routing"):
            agent_routing = str(payload["agent_routing"]).strip()

    executor_model = _resolve_model("EXECUTOR_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    optimizer_model = _resolve_model("OPTIMIZER_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    print(
        f"start planner optimization: backend={args.backend} objective={objective} "
        f"slots={','.join(slots)} train={args.train_split} dev={args.dev_split} "
        f"max_steps={args.max_steps} rollback={not args.no_rollback}"
    )
    if objective == "e2e":
        print(f"e2e_profile={args.e2e_profile} e2e_timeout={args.e2e_timeout}")
    print(f"executor_model={executor_model} optimizer_model={optimizer_model}")

    output = await run_planner_optimization(
        backend=args.backend,
        slots=slots,
        decomposition_prompt=decomposition_prompt,
        agent_routing=agent_routing,
        registry=registry,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=args.max_steps,
        failure_threshold=args.failure_threshold,
        rollback=not args.no_rollback,
        train_split=args.train_split,
        dev_split=args.dev_split,
        objective=objective,
        e2e_profile=args.e2e_profile,
        e2e_timeout_sec=args.e2e_timeout,
        enable_guess_agent=not args.no_guess_agent,
        max_failure_cases_per_step=args.max_failure_cases,
    )

    report_suffix = args.backend
    if objective == "e2e":
        report_suffix = f"{args.backend}_e2e"
    output_path = args.output or optimized_prompts_path(fixtures.locale)
    report_path = args.report_output or output_path.parent / f"planner_{report_suffix}_optimization_report.json"

    save_planner_optimization_artifacts(
        locale=fixtures.locale,
        output_path=output_path,
        report_path=report_path,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        backend=args.backend,
        decomposition_result=output.decomposition_result,
        routing_result=output.routing_result,
    )
    _print_slot_result("decomposition", output.decomposition_result)
    _print_slot_result("routing", output.routing_result)
    print(f"saved prompt: {output_path}")
    print(f"saved report: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel Planner 一键优化（decomposition → routing）")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--prompt-file", type=Path, help="起始 prompt JSON（可含两个字段）")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "textgrad_lib", "textgrad_graph"],
        help="local=自写 TextGrad 风格；textgrad_lib=失败文本+TGD；textgrad_graph=三步计算图反传",
    )
    parser.add_argument(
        "--slots",
        default="all",
        help="all 或 decomposition,routing 的逗号列表",
    )
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "test", "all"])
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument(
        "--objective",
        default="l1_l2",
        choices=["l1_l2", "e2e"],
        help="l1_l2=B1 planner 分；e2e=B2 端到端 graph loss + E2E rollback（需 textgrad_graph）",
    )
    parser.add_argument(
        "--e2e-profile",
        default="workflow",
        choices=["workflow", "legacy"],
        help="E2E 编排路径（objective=e2e 时生效）",
    )
    parser.add_argument(
        "--e2e-timeout",
        type=float,
        default=None,
        help="单条 E2E case 超时秒数（objective=e2e 时生效）",
    )
    parser.add_argument(
        "--max-failure-cases",
        type=int,
        default=3,
        help="每步 TextGrad backward 最多使用的 train 失败 case 数（objective=e2e，按分数从低到高）",
    )
    parser.add_argument(
        "--no-guess-agent",
        action="store_true",
        help="E2E 评测时关闭 guess_agent 回退",
    )
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
