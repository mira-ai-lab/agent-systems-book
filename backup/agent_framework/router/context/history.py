"""对话历史格式化（供 history_gate / interaction_rewrite 消费）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

HistoryInput = Union[str, Sequence["HistoryTurn"], None]


@dataclass(frozen=True)
class HistoryTurn:
    query: str
    response: str


def format_history_text(history: HistoryInput) -> str:
    if history is None:
        return ""
    if isinstance(history, str):
        return history.strip()
    lines: List[str] = []
    for idx, turn in enumerate(history, start=1):
        if isinstance(turn, HistoryTurn):
            q, r = turn.query, turn.response
        elif isinstance(turn, dict):
            q = str(turn.get("query") or turn.get("his_query") or "")
            r = str(turn.get("response") or turn.get("his_response") or "")
        else:
            continue
        lines.append(f"第{idx}轮:")
        lines.append(f"用户: {q.strip()}")
        lines.append(f"助手: {r.strip()}")
    return "\n".join(lines).strip()


def normalize_history(history: Optional[HistoryInput]) -> str:
    text = format_history_text(history)
    return text if text else ""
