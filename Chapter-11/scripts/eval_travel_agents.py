#!/usr/bin/env python3
"""Travel 子 Agent 单节点 benchmark 评测。

用法::

    python scripts/eval_travel_agents.py --agent FlightAgent --split dev
    python scripts/eval_travel_agents.py --agent all --split dev
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
from agent_framework.optimization.agent_pipeline import parse_agent_slots
from agent_framework.optimization.agents.evaluator import (
    create_agent_bridge,
    evaluate_single_agent_benchmark,
)
from agent_framework.optimization.agents.fixtures import load_single_agent_cases
from agent_framework.optimization.agents.runtime import default_agent_prompt_template


def _resolve_model() -> str:
    return (
        os.getenv("EXECUTOR_MODEL", "").strip()
        or os.getenv("DASHSCOPE_CHAT_MODEL", "").strip()
        or "qwen-plus"
    )


def _print_report(report) -> None:
    print(
        f"agent={report.agent_name} locale={report.locale} split={report.split} "
        f"cases={report.case_count} avg={report.average_score:.3f}"
    )
    for item in report.cases:
        status = "PASS" if item.score.total >= 0.8 else "FAIL"
        print(
            f"[{status}] {item.case_id} score={item.score.total:.3f} "
            f"tools={item.invoked_tools}"
        )
        if item.score.details:
            print(f"  details: {'; '.join(item.score.details)}")


async def _run(args: argparse.Namespace) -> int:
    load_project_dotenv()
    fixtures = load_single_agent_cases(args.fixtures)
    agents = parse_agent_slots(args.agent)
    llm = create_llm(temperature=0, model=_resolve_model())

    all_reports = []
    for agent_name in agents:
        bridge = create_agent_bridge(llm, agent_name=agent_name, locale=fixtures.locale)
        template = default_agent_prompt_template(agent_name, locale=fixtures.locale)

        report = await evaluate_single_agent_benchmark(
            bridge,
            fixtures=fixtures,
            agent_name=agent_name,
            split=args.split,
            system_prompt_template=template,
        )
        all_reports.append(report)

        if args.json and len(agents) == 1:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        elif not args.json:
            _print_report(report)
            if len(agents) > 1:
                print()

    if args.json and len(agents) > 1:
        print(json.dumps([r.to_dict() for r in all_reports], ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = all_reports[0].to_dict() if len(all_reports) == 1 else [r.to_dict() for r in all_reports]
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel 子 Agent benchmark 评测")
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument(
        "--agent",
        default="FlightAgent",
        help="Agent 名，或 all，或逗号分隔列表",
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
