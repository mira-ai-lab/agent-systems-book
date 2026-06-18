"""Server-Sent Events 格式化。"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterable, Iterator

from agent_framework.stream.events import public_event


def format_sse(event: Dict[str, Any], *, event_id: str | None = None) -> str:
    payload = public_event(event)
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {payload.get('type', 'message')}")
    lines.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def iter_sse(events: Iterable[Dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield format_sse(event)


async def async_iter_sse(events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield format_sse(event)
