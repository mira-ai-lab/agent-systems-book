#!/usr/bin/env python3
"""KB 召回评测：hashing vs embedding hit@k 对比报告。

用法::

    python scripts/benchmark_knowledge_recall.py
    python scripts/benchmark_knowledge_recall.py --backends hashing,embedding --json
    python scripts/benchmark_knowledge_recall.py --fixtures data/knowledge/benchmark/fixtures.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import load_project_dotenv
from agent_framework.router.kb.benchmark import (
    compare_backend_recall,
    default_fixtures_path,
    load_benchmark_fixtures,
)


def _parse_backends(raw: str) -> list[str]:
    backends = [item.strip().lower() for item in (raw or "").split(",") if item.strip()]
    if not backends:
        raise ValueError("backends 不能为空")
    return backends


def _parse_top_k(raw: str) -> list[int]:
    values = [int(item.strip()) for item in (raw or "").split(",") if item.strip()]
    if not values:
        raise ValueError("top-k 不能为空")
    return values


def _embedding_available() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))


def main() -> int:
    parser = argparse.ArgumentParser(description="KB 召回评测（hashing vs embedding）")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=default_fixtures_path(),
        help="benchmark fixture JSON 路径",
    )
    parser.add_argument(
        "--backends",
        default="hashing",
        help="逗号分隔：hashing,embedding",
    )
    parser.add_argument(
        "--top-k",
        default="1,3,5",
        help="hit@k 指标，逗号分隔",
    )
    parser.add_argument(
        "--vector-min-score",
        type=float,
        default=0.01,
        help="向量 raw_score 下限（评测时放宽阈值）",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="在默认 backends 基础上强制包含 embedding",
    )
    parser.add_argument("--json", action="store_true", help="JSON 报告输出到 stdout")
    parser.add_argument("--output", type=Path, help="JSON 报告写入文件")
    args = parser.parse_args()

    load_project_dotenv()

    domain, cases = load_benchmark_fixtures(args.fixtures)
    backends = _parse_backends(args.backends)
    if args.include_embedding and "embedding" not in backends:
        backends.append("embedding")

    filtered: list[str] = []
    skipped: list[str] = []
    for backend in backends:
        if backend == "embedding" and not _embedding_available():
            skipped.append("embedding (missing DASHSCOPE_API_KEY / OPENAI_API_KEY)")
            continue
        filtered.append(backend)
    if not filtered:
        print("[error] 无可用 backend；hashing 始终可用，embedding 需 API Key", file=sys.stderr)
        return 1

    top_k_values = _parse_top_k(args.top_k)
    report = compare_backend_recall(
        domain,
        cases,
        backends=filtered,
        top_k_values=top_k_values,
        vector_min_score=args.vector_min_score,
    )
    if skipped:
        report["skipped_backends"] = skipped

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json or args.output:
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if args.json:
            print(payload)
    else:
        print(f"KB Recall Benchmark — domain={domain}, cases={len(cases)}")
        print("=" * 60)
        for backend, payload in report["backends"].items():
            print(f"[{backend}]")
            for metric, value in payload["hit_at_k"].items():
                print(f"  {metric}: {value:.2%}")
            for case in payload["cases"]:
                status = "HIT" if case["hit"] else "MISS"
                rank = case["rank"] or "-"
                top = case.get("top_match") or {}
                preview = top.get("doc_id") or top.get("name") or "-"
                print(f"  - {status} @{rank} {case['case_id']}: {case['query'][:40]} -> {preview}")
            print("-" * 60)
        winner = report.get("comparison", {}).get("winners", {})
        if winner:
            metric, name = next(iter(winner.items()))
            print(f"Winner ({metric}): {name}")
        if skipped:
            print("Skipped:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
