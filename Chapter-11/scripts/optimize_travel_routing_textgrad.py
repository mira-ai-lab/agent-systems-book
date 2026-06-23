#!/usr/bin/env python3
"""Travel agent_routing prompt 优化（textgrad 库版 Optimizer）。

用法::

    pip install -e ".[evolution]"
    python scripts/optimize_travel_routing_textgrad.py --max-steps 3
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
from agent_framework.optimization.core.save import save_routing_optimization_artifacts
from agent_framework.optimization.decomposition.fixtures import default_fixtures_path, load_decomposition_fixtures
from agent_framework.optimization.optimizers.textgrad_lib.routing import optimize_agent_routing_prompt_textgrad
from agent_framework.optimization.prompt_store import optimized_prompts_path
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


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
    base_prompts = TravelPrompts.build(locale=fixtures.locale, use_optimized=False)
    agent_routing = base_prompts.agent_routing
    decomposition_prompt = None

    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        agent_routing = str(payload.get("agent_routing") or agent_routing).strip()
        if payload.get("decomposition_prompt"):
            decomposition_prompt = str(payload["decomposition_prompt"]).strip()

    executor_model = _resolve_model("EXECUTOR_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    optimizer_model = _resolve_model("OPTIMIZER_MODEL", "DASHSCOPE_CHAT_MODEL", "qwen-plus")
    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    print(
        f"start textgrad_lib routing optimization: train={args.train_split} dev={args.dev_split} "
        f"max_steps={args.max_steps}"
    )

    result = await optimize_agent_routing_prompt_textgrad(
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
        decomposition_prompt=decomposition_prompt,
    )

    output_path = args.output or optimized_prompts_path(fixtures.locale)
    report_path = args.report_output or output_path.parent / "routing_textgrad_optimization_report.json"

    save_routing_optimization_artifacts(
        result,
        locale=fixtures.locale,
        output_path=output_path,
        report_path=report_path,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        decomposition_prompt=decomposition_prompt,
        extra_metadata={"optimizer_backend": "textgrad_lib"},
    )

    print(
        f"baseline_dev={result.baseline_dev_score:.3f} best_dev={result.best_dev_score:.3f} "
        f"optimizer={result.optimizer} saved={output_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel agent_routing prompt 优化（textgrad 库）")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
