#!/usr/bin/env python3
"""Travel 任务拆解 prompt 优化（textgrad 库版 Optimizer）。

与 ``optimize_travel_decomposition.py``（local optimizer）并存：
  - local: 规则反馈 + 自写 PROMPT_REVISION_TEMPLATE
  - textgrad_lib: ``textgrad.Variable`` + ``TextualGradientDescent``

用法::

    pip install -e ".[evolution]"
    python scripts/optimize_travel_decomposition_textgrad.py --max-steps 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import create_llm, load_project_dotenv
from agent_framework.optimization.core.save import save_decomposition_optimization_artifacts
from agent_framework.optimization.decomposition.fixtures import (
    default_fixtures_path,
    load_decomposition_fixtures,
)
from agent_framework.optimization.optimizers.textgrad_lib.decomposition import (
    optimize_decomposition_prompt_textgrad,
)
from agent_framework.optimization.prompt_store import optimized_prompts_path
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub

DEFAULT_REPORT_SUFFIX = "decomposition_textgrad_optimization_report.json"


def _resolve_model(env_name: str, fallback_env: str, default: str) -> str:
    return (
        os.getenv(env_name, "").strip()
        or os.getenv(fallback_env, "").strip()
        or default
    )


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()

    fixtures = load_decomposition_fixtures(args.fixtures)
    registry = create_travel_registry_stub()
    base_prompt = TravelPrompts.build(locale=fixtures.locale, use_optimized=False).decomposition_prompt

    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        custom_prompt = str(payload.get("decomposition_prompt") or "").strip()
        if not custom_prompt:
            raise ValueError(f"{args.prompt_file} 缺少 decomposition_prompt")
        base_prompt = custom_prompt

    executor_model = _resolve_model("EXECUTOR_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    optimizer_model = _resolve_model("OPTIMIZER_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")

    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    print(
        f"start textgrad_lib optimization: train={args.train_split} dev={args.dev_split} "
        f"max_steps={args.max_steps} rollback={not args.no_rollback}"
    )
    print(f"executor_model={executor_model} optimizer_model={optimizer_model}")

    result = await optimize_decomposition_prompt_textgrad(
        decomposition_prompt=base_prompt,
        registry=registry,
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        fixtures=fixtures,
        max_steps=args.max_steps,
        failure_threshold=args.failure_threshold,
        rollback=not args.no_rollback,
        train_split=args.train_split,
        dev_split=args.dev_split,
    )

    output_path = args.output or optimized_prompts_path(fixtures.locale)
    report_path = args.report_output or output_path.parent / DEFAULT_REPORT_SUFFIX

    save_decomposition_optimization_artifacts(
        result,
        locale=fixtures.locale,
        output_path=output_path,
        report_path=report_path,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        extra_metadata={"optimizer_backend": "textgrad_lib"},
    )

    print(
        f"baseline_dev={result.baseline_dev_score:.3f} "
        f"best_dev={result.best_dev_score:.3f} steps={len(result.steps)} "
        f"optimizer={result.optimizer}"
    )
    for step in result.steps:
        flag = "ACCEPT" if step.accepted else "REJECT"
        print(
            f"  step={step.step} {flag} train={step.train_average:.3f} "
            f"candidate_dev={step.candidate_dev_average:.3f} best_dev={step.dev_average:.3f} "
            f"failures={step.failure_count}"
        )
    print(f"saved prompt: {output_path}")
    print(f"saved report: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel decomposition prompt 优化（textgrad 库）")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--prompt-file", type=Path, help="可选：起始 decomposition_prompt JSON")
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "test", "all"])
    parser.add_argument("--dev-split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--output", type=Path, help="优化后 prompt JSON")
    parser.add_argument("--report-output", type=Path, help="优化过程报告路径")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
