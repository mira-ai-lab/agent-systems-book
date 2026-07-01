"""子任务分层执行：WeatherAgent 串行（避免 MCP stdio 并发冲突）。"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List

# MCP / npx stdio 单会话，同层多个 WeatherAgent 并行会导致 unavailable
SERIAL_AGENTS = frozenset({"WeatherAgent", "HotelAgent"})


async def run_task_layer(
    layer: List[str],
    subtasks: Dict[str, Dict[str, Any]],
    invoke: Callable[[str], Awaitable[Any]],
) -> Dict[str, Any]:
    """执行同层子任务；WeatherAgent 串行，其余可 asyncio.gather 并行。"""
    results: Dict[str, Any] = {}
    serial = [tid for tid in layer if subtasks.get(tid, {}).get("agent") in SERIAL_AGENTS]
    parallel = [tid for tid in layer if tid not in serial]

    for tid in serial:
        results[tid] = await invoke(tid)

    if len(parallel) == 1:
        tid = parallel[0]
        results[tid] = await invoke(tid)
    elif parallel:
        layer_results = await asyncio.gather(*[invoke(tid) for tid in parallel])
        for tid, res in zip(parallel, layer_results):
            results[tid] = res

    return results
