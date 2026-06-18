"""Supervisor stage 级累计压缩（对齐 router-sdk task.stage）。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.orchestration.supervisor.step_summary import extract_handoff_contexts
from agent_framework.router.prompts.loader import get_stage_summary_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


def build_multi_step_context(
    subtask_results: Dict[str, Any],
    contexts: Sequence[Dict[str, str]],
    *,
    locale: str = "zh",
) -> str:
    prompts = get_stage_summary_prompts(locale)
    desc_tpl = prompts.get("step_level_context_desc", "")
    detail_tpl = prompts.get("step_level_context_detail", "")
    blocks: List[str] = []
    for (tid, info), ctx in zip(subtask_results.items(), contexts):
        step_desc = str(ctx.get("agent_query") or info.get("agent") or tid)
        blocks.append(desc_tpl.format(step_desc=step_desc))
        blocks.append(
            detail_tpl.format(
                agent_name=str(ctx.get("agent_name") or info.get("agent") or ""),
                query=str(ctx.get("agent_query") or ""),
                response=str(info.get("step_summary") or info.get("agent_summary") or ""),
            )
        )
    return "\n".join(blocks).strip()


def _current_step_label(subtask_results: Dict[str, Any]) -> str:
    if not subtask_results:
        return "无"
    last_tid, last_info = list(subtask_results.items())[-1]
    agent = str(last_info.get("agent") or "")
    return f"{last_tid} / {agent}".strip(" /")


@trace_span(
    name=span_name("supervisor.stage_summary"),
    attrs_args=["global_query"],
    record_result=False,
)
async def run_stage_summary(
    llm: ChatOpenAI,
    *,
    global_query: str,
    multi_step_context: str,
    task_status: str = "completed",
    last_stage_summary: str = "",
    current_step: str = "",
    next_step_desc: str = "",
    locale: str = "zh",
) -> str:
    prompts = get_stage_summary_prompts(locale)
    prompt = prompts.get("prompt", "").format(
        global_query=global_query.strip() or "未指定",
        task_status=task_status.strip() or "completed",
        last_stage_summary=last_stage_summary.strip() or prompts.get("no_previous_summary", ""),
        current_step=current_step.strip() or "无",
        multi_step_context=multi_step_context.strip(),
        next_step_desc=next_step_desc.strip() or prompts.get("all_steps_completed", ""),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return str(response.content or "").strip()


class StageSummarizer:
    """对一个阶段内多个 step summary 做累计压缩。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        locale: str = "zh",
        min_steps: int = 2,
    ) -> None:
        self.llm = llm
        self.locale = locale
        self.min_steps = max(1, min_steps)

    async def summarize_stage(
        self,
        *,
        user_query: str,
        messages: Sequence[Any],
        subtask_results: Dict[str, Any],
        handoff_node_names: set[str],
        last_stage_summary: str = "",
    ) -> str:
        if len(subtask_results) < self.min_steps:
            return ""
        contexts = extract_handoff_contexts(messages, handoff_node_names)
        if not contexts:
            return ""
        multi_step = build_multi_step_context(subtask_results, contexts, locale=self.locale)
        return await run_stage_summary(
            self.llm,
            global_query=user_query,
            multi_step_context=multi_step,
            last_stage_summary=last_stage_summary,
            current_step=_current_step_label(subtask_results),
            locale=self.locale,
        )
