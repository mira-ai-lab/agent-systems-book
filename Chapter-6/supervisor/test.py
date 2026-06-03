"""Supervisor 模式演示"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SUP_DIR = Path(__file__).resolve().parent
if str(SUP_DIR) not in sys.path:
    sys.path.insert(0, str(SUP_DIR))

import bootstrap  # noqa: E402

bootstrap.setup()

from orchestrator import SupervisorOrchestrator  # noqa: E402


async def main() -> None:
    # 切换长期记忆后端: "chroma" | "store"
    # 也可设环境变量 MEMORY_BACKEND=store
    backend = os.getenv("MEMORY_BACKEND", "chroma")

    orchestrator = SupervisorOrchestrator(
        enable_memory=False,
        long_term_backend=backend,
    )

    print(f"\n长期记忆后端: {orchestrator.long_term_backend}")

    result = await orchestrator.process_request(
        "查询上海2026年6月2号天气",
        thread_id="supervisor_demo",
    )
    print(f"\n最终回复长度: {len(result.get('final_response') or '')} 字符")
    print(result["final_response"][:800])


if __name__ == "__main__":
    asyncio.run(main())
