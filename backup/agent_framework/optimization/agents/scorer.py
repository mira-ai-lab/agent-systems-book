"""子 Agent 单节点 benchmark 规则打分。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .fixtures import SingleAgentCase


@dataclass
class SingleAgentScore:
    """单 Agent 一次 invoke 的得分。"""

    total: float
    response_ok: bool
    keyword_ok: bool
    tool_called_ok: bool
    invoked_tools: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "response_ok": self.response_ok,
            "keyword_ok": self.keyword_ok,
            "tool_called_ok": self.tool_called_ok,
            "invoked_tools": list(self.invoked_tools),
            "details": list(self.details),
        }


def extract_ai_text(state: Dict[str, Any]) -> str:
    """从 LangGraph Agent 状态中提取最终 AI 回复文本。"""
    messages = state.get("messages") or []
    chunks: List[str] = []
    for msg in messages:
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            chunks.append(str(msg.content))
    return "\n".join(chunks).strip()


def extract_invoked_tool_names(state: Dict[str, Any]) -> List[str]:
    """从消息流中收集被调用过的工具名。"""
    names: Set[str] = set()
    for msg in state.get("messages") or []:
        if getattr(msg, "type", None) != "tool":
            continue
        name = getattr(msg, "name", None)
        if name:
            names.add(str(name))
    return sorted(names)


def score_single_agent_run(
    state: Dict[str, Any],
    case: SingleAgentCase,
) -> SingleAgentScore:
    """对 Agent invoke 结果按 fixture 期望打分。"""
    details: List[str] = []
    total = 0.0

    final_text = extract_ai_text(state)
    invoked_tools = extract_invoked_tool_names(state)

    response_ok = bool(final_text)
    if response_ok:
        total += 0.25
    else:
        details.append("缺少 AI 回复文本")

    tool_called_ok = case.tool in invoked_tools
    if tool_called_ok:
        total += 0.35
    else:
        details.append(f"未调用期望工具 {case.tool}，实际: {invoked_tools or '(none)'}")

    missing_keywords = [kw for kw in case.response_keywords if kw not in final_text]
    keyword_ok = not missing_keywords
    if keyword_ok:
        total += 0.40
    else:
        ratio = (
            (len(case.response_keywords) - len(missing_keywords)) / len(case.response_keywords)
            if case.response_keywords
            else 1.0
        )
        total += 0.40 * max(0.0, ratio)
        details.append(f"回复缺少关键词: {missing_keywords}")

    return SingleAgentScore(
        total=min(total, 1.0),
        response_ok=response_ok,
        keyword_ok=keyword_ok,
        tool_called_ok=tool_called_ok,
        invoked_tools=invoked_tools,
        details=details,
    )
