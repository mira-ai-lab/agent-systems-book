#!/usr/bin/env python3
"""Travel decomposition prompt optimizer (textgrad graph / Phase B1)."""

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
from agent_framework.optimization.core.save import save_decomposition_optimization_artifacts
from agent_framework.optimization.decomposition.fixtures import default_fixtures_path, load_decomposition_fixtures
from agent_framework.optimization.optimizers.textgrad_graph.decomposition import (
    optimize_decomposition_prompt_graph,
)
from agent_framework.optimization.prompt_store import optimized_prompts_path
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def main() -> int:
    load_project_dotenv()
    parser = argparse.ArgumentParser(description="Travel decomposition graph optimizer (Phase B1)")
    parser.add_argument("--fixtures", type=Path, default=default_fixtures_path())
    parser.add_argument("--train-split", default="dev")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--no-rollback", action="store_true")
    args = parser.parse_args()

    fixtures = load_decomposition_fixtures(args.fixtures)
    registry = create_travel_registry_stub()
    prompts = TravelPrompts.build(locale=fixtures.locale, use_optimized=False)
    executor_model = os.getenv("EXECUTOR_MODEL") or os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen-plus"
    optimizer_model = os.getenv("OPTIMIZER_MODEL") or executor_model
    executor_llm = create_llm(temperature=0, model=executor_model)
    optimizer_llm = create_llm(temperature=0.2, model=optimizer_model)

    result = asyncio.run(
        optimize_decomposition_prompt_graph(
            decomposition_prompt=prompts.decomposition_prompt,
            agent_routing=prompts.agent_routing,
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
    )

    output_path = optimized_prompts_path(fixtures.locale)
    report_path = output_path.parent / "decomposition_textgrad_graph_optimization_report.json"
    save_decomposition_optimization_artifacts(
        result,
        locale=fixtures.locale,
        output_path=output_path,
        report_path=report_path,
        executor_model=executor_model,
        optimizer_model=optimizer_model,
        extra_metadata={"backend": "textgrad_graph"},
    )
    print(f"baseline_dev={result.baseline_dev_score:.3f} best_dev={result.best_dev_score:.3f}")
    print(f"saved prompt: {output_path}")
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
