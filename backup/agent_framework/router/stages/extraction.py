"""事件抽取（selection.extraction）。"""

from __future__ import annotations

import ast
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.domain.parsing import parse_json_from_llm
from agent_framework.router.prompts.loader import get_extraction_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


def parse_extraction_response(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        parsed = parse_json_from_llm(raw)
    except (ValueError, SyntaxError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw] if raw else []

    if isinstance(parsed, list):
        if len(parsed) == 3 and isinstance(parsed[2], list):
            return _normalize_events(parsed[2])
        return _normalize_events(parsed)
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    return []


def _normalize_events(items: list) -> List[str]:
    events: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            events.append(text)
    return events


@trace_span(
    name=span_name("router.extraction"),
    attrs_args=["query"],
    record_result=False,
)
async def run_extraction(
    llm: ChatOpenAI,
    query: str,
    *,
    locale: str = "zh",
    history: Optional[str] = None,
    note: str = "",
) -> List[str]:
    prompts = get_extraction_prompts(locale)
    if history and history.strip():
        template = prompts.get("multi", "")
        prompt = template.format(note=note or "", history=history.strip(), query=query.strip())
    else:
        template = prompts.get("single", "")
        prompt = template.format(query=query.strip())
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return parse_extraction_response(str(response.content or ""))
