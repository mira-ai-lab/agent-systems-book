"""InstructionBuilder：handoff / 子任务执行前的指令构建。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent_framework.router.prompts.loader import get_instruction_build_prompts
from agent_framework.tracing.trace_provider import span_name, trace_span


@trace_span(
    name=span_name("router.instruction_build"),
    attrs_args=["init_task", "target_agent"],
    record_result=False,
)
async def run_instruction_build(
    llm: ChatOpenAI,
    *,
    init_task: str,
    target_agent: str,
    agent_skill: str,
    previous_step_info: str = "",
    locale: str = "zh",
) -> str:
    prompts = get_instruction_build_prompts(locale)
    system = prompts.get("system", "")
    user = prompts.get("user_template", "").format(
        init_task=init_task.strip(),
        target_agent=target_agent.strip(),
        agent_skill=agent_skill.strip() or "未提供",
        previous_step_info=previous_step_info.strip() or "无",
    )
    response = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    instruction = str(response.content or "").strip()
    return instruction or init_task.strip()


class InstructionBuilder:
    """对 `run_instruction_build` 的轻量封装，供编排层复用。"""

    def __init__(self, llm: ChatOpenAI, *, locale: str = "zh") -> None:
        self.llm = llm
        self.locale = locale

    async def build(
        self,
        *,
        init_task: str,
        target_agent: str,
        agent_skill: str,
        previous_step_info: str = "",
    ) -> str:
        return await run_instruction_build(
            self.llm,
            init_task=init_task,
            target_agent=target_agent,
            agent_skill=agent_skill,
            previous_step_info=previous_step_info,
            locale=self.locale,
        )
