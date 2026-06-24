#!/usr/bin/env python3
"""Travel mini-pipeline benchmark 评测（Agent-B3）。

用法::

    python scripts/eval_travel_agent_pipeline.py --split dev
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
from agent_framework.optimization.agent_mini_pipeline import _build_initial_prompt_templates
from agent_framework.optimization.agents.mini_pipeline.evaluator import evaluate_mini_pipeline_benchmark
from agent_framework.optimization.agents.mini_pipeline.fixtures import load_mini_pipeline_cases
from agent_framework.optimization.agents.mini_pipeline.runtime import MiniPipelineRunner


def _resolve_model() -> str:
    return (
        os.getenv("EXECUTOR_MODEL", "").strip()
        or os.getenv("DASHSCOPE_CHAT_MODEL", "").strip()
        or "qwen-plus"
    )


def _print_report(report) -> None:
    print(
        f"mini-pipeline locale={report.locale} split={report.split} "
        f"cases={report.case_count} avg={report.average_score:.3f}"
    )
    for item in report.cases:
        status = "PASS" if item.score.total >= 0.8 else "FAIL"
        print(
            f"[{status}] {item.case_id} score={item.score.total:.3f} "
            f"agents={item.score.invoked_agents} completed={item.score.completed_steps}"
        )
        if item.score.details:
            print(f"  details: {'; '.join(item.score.details)}")


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()
    fixtures = load_mini_pipeline_cases(args.fixtures)
    llm = create_llm(temperature=0, model=_resolve_model())
    runner = MiniPipelineRunner(llm=llm, locale=fixtures.locale)
    templates = _build_initial_prompt_templates(fixtures)

    report = await evaluate_mini_pipeline_benchmark(
        runner,
        fixtures=fixtures,
        split=args.split,
        prompt_templates=templates,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel mini-pipeline benchmark 评测")
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
