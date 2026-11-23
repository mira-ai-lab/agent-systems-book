"""interaction_rewrite：多轮槽位补全与 query 重写。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent_framework.router.prompts.loader import get_interaction_rewrite_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


@trace_span(
    name=span_name("router.interaction_rewrite"),
    attrs_args=["query"],
    record_result=False,
)
async def run_interaction_rewrite(
    llm: ChatOpenAI,
    query: str,
    history: str,
    *,
    locale: str = "zh",
    task_info: str = "",
) -> str:
    prompts = get_interaction_rewrite_prompts(locale)
    system = prompts.get("system", "").format(task_info=task_info or "未指定")
    user = prompts.get("user_template", "").format(
        history=history.strip(),
        query=query.strip(),
    )
    response = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    rewritten = str(response.content or "").strip()
    return rewritten or query.strip()
