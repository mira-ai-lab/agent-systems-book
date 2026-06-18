"""Supervisor step 级上下文压缩（对齐 router-sdk task.summary）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.router.prompts.loader import get_step_summary_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


def format_round_info(
    *,
    idx: int,
    agent_name: str,
    agent_query: str,
    agent_response: str,
    locale: str = "zh",
) -> str:
    prompts = get_step_summary_prompts(locale)
    template = prompts.get("round_info", "")
    return template.format(
        idx=idx,
        agent_name=agent_name,
        agent_query=agent_query.strip(),
        agent_response=agent_response.strip(),
    )


def build_multi_round_context(rounds: Sequence[Dict[str, str]], *, locale: str = "zh") -> str:
    blocks: List[str] = []
    for idx, item in enumerate(rounds, start=1):
        blocks.append(
            format_round_info(
                idx=idx,
                agent_name=str(item.get("agent_name") or ""),
                agent_query=str(item.get("agent_query") or ""),
                agent_response=str(item.get("agent_response") or ""),
                locale=locale,
            )
        )
    return "\n".join(blocks).strip()


def extract_handoff_contexts(
    messages: Sequence[Any],
    handoff_node_names: set[str],
) -> List[Dict[str, str]]:
    """从 Supervisor 消息流提取每次 handoff 的 query / response。"""
    contexts: List[Dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage) or msg.name not in handoff_node_names:
            continue
        agent_query = _find_agent_query(messages[:i])
        contexts.append(
            {
                "agent_name": str(msg.name),
                "agent_query": agent_query,
                "agent_response": str(msg.content or ""),
            }
        )
    return contexts


def _find_agent_query(prior_messages: Sequence[Any]) -> str:
    for prev in reversed(prior_messages):
        tool_calls = getattr(prev, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                if isinstance(args, dict):
                    for key in ("task", "instructions", "instruction", "query"):
                        value = str(args.get(key) or "").strip()
                        if value:
                            return value
        if isinstance(prev, HumanMessage):
            text = str(prev.content or "").strip()
            if text:
                return text[:800]
    return ""


@trace_span(
    name=span_name("supervisor.step_summary"),
    attrs_args=["agent_name", "step_desc"],
    record_result=False,
)
async def run_step_summary(
    llm: ChatOpenAI,
    *,
    step_desc: str,
    agent_name: str,
    multi_round_context: str,
    step_status: str = "completed",
    next_step_desc: str = "所有步骤已完成，无下一个子步骤描述",
    locale: str = "zh",
) -> str:
    prompts = get_step_summary_prompts(locale)
    prompt = prompts.get("prompt", "").format(
        step_desc=step_desc.strip() or "未指定",
        agent_name=agent_name.strip(),
        step_status=step_status.strip() or "completed",
        multi_round_context=multi_round_context.strip(),
        next_step_desc=next_step_desc.strip(),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    summary = str(response.content or "").strip()
    return summary


class StepSummarizer:
    """压缩 Supervisor 单次 handoff 的多轮交互，供后续步骤引用。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        locale: str = "zh",
        min_chars: int = 200,
    ) -> None:
        self.llm = llm
        self.locale = locale
        self.min_chars = max(0, min_chars)

    async def summarize_handoff(
        self,
        *,
        agent_name: str,
        agent_query: str,
        agent_response: str,
        step_desc: str = "",
        step_status: str = "completed",
        next_step_desc: str = "",
    ) -> str:
        if len(agent_response.strip()) < self.min_chars:
            return agent_response.strip()
        multi_round = build_multi_round_context(
            [
                {
                    "agent_name": agent_name,
                    "agent_query": agent_query,
                    "agent_response": agent_response,
                }
            ],
            locale=self.locale,
        )
        return await run_step_summary(
            self.llm,
            step_desc=step_desc or agent_query or agent_name,
            agent_name=agent_name,
            multi_round_context=multi_round,
            step_status=step_status,
            next_step_desc=next_step_desc or "所有步骤已完成，无下一个子步骤描述",
            locale=self.locale,
        )

    async def summarize_subtask_results(
        self,
        messages: Sequence[Any],
        subtask_results: Dict[str, Any],
        handoff_node_names: set[str],
        *,
        user_query: str = "",
    ) -> Dict[str, Any]:
        contexts = extract_handoff_contexts(messages, handoff_node_names)
        if not contexts:
            return subtask_results

        ordered = list(subtask_results.items())
        updated: Dict[str, Any] = {}
        for (tid, info), ctx in zip(ordered, contexts):
            summary = await self.summarize_handoff(
                agent_name=str(ctx.get("agent_name") or info.get("agent") or ""),
                agent_query=str(ctx.get("agent_query") or user_query),
                agent_response=str(ctx.get("agent_response") or info.get("agent_summary") or ""),
                step_desc=str(ctx.get("agent_query") or user_query)[:200],
                next_step_desc=_next_step_desc(ordered, tid),
            )
            merged = dict(info)
            merged["agent_summary"] = summary
            merged["step_summary"] = summary
            updated[tid] = merged
        for tid, info in ordered[len(contexts) :]:
            updated[tid] = info
        return updated


def _next_step_desc(ordered_items: Sequence[tuple[str, Any]], current_tid: str) -> str:
    ids = [tid for tid, _ in ordered_items]
    if current_tid not in ids:
        return "所有步骤已完成，无下一个子步骤描述"
    idx = ids.index(current_tid)
    if idx + 1 >= len(ids):
        return "所有步骤已完成，无下一个子步骤描述"
    _next_tid, next_info = ordered_items[idx + 1]
    agent = str(next_info.get("agent") or "")
    summary = str(next_info.get("agent_summary") or "")[:120]
    return f"下一子步骤由 {agent} 执行：{summary}".strip()
