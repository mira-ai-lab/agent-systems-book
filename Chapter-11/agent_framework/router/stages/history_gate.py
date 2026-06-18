"""history_gate：判断对话历史与当前输入是否相关（0/1）。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.router.prompts.loader import get_history_gate_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


def parse_history_gate_response(text: str) -> bool:
    core = (text or "").strip().replace(" ", "")
    if core == "0":
        return False
    if core == "1" or core.endswith("1"):
        return True
    return "1" in core


@trace_span(
    name=span_name("router.history_gate"),
    attrs_args=["query"],
    record_result=False,
)
async def run_history_gate(
    llm: ChatOpenAI,
    query: str,
    history: str,
    *,
    locale: str = "zh",
) -> bool:
    prompts = get_history_gate_prompts(locale)
    template = prompts.get("history_prompt", "")
    prompt = template.format(history=history.strip(), query=query.strip())
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return parse_history_gate_response(str(response.content or ""))
