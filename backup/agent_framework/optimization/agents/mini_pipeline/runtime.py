"""Mini-pipeline 运行时：固定 subtask 依次调用各子 Agent（不经过 LangGraph）。"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agents.runtime import AgentSyncBridge
from agent_framework.optimization.agents.scorer import (
    extract_ai_text,
    extract_invoked_tool_names,
    score_single_agent_run,
)

from .fixtures import MiniPipelineCase


class MiniPipelineRunner:
    """按 fixture steps 顺序 invoke 子 Agent，汇总为 pipeline 结果 dict。"""

    def __init__(self, *, llm: ChatOpenAI, locale: str = "zh"):
        self._llm = llm
        self._locale = locale

    def run_case(
        self,
        case: MiniPipelineCase,
        *,
        prompt_templates: Mapping[str, str],
        thread_id_prefix: str = "",
    ) -> Dict[str, Any]:
        """同步执行一条 mini-pipeline case。"""
        step_results: Dict[str, Any] = {}
        response_parts: list[str] = []
        min_step_score = case.expect.min_step_score

        for step in case.steps:
            template = prompt_templates.get(step.agent_name)
            if not template:
                raise ValueError(f"缺少 {step.agent_name} 的 prompt_templates")

            bridge = AgentSyncBridge(
                llm=self._llm,
                locale=self._locale,
                agent_name=step.agent_name,
            )
            state = bridge.invoke(
                system_prompt_template=template,
                user_query=step.subtask,
                thread_id=f"{thread_id_prefix}{case.case_id}-{step.step_id}",
            )
            single_case = step.to_single_agent_case(case_id=case.case_id)
            step_score = score_single_agent_run(state, single_case)
            raw_response = extract_ai_text(state)
            invoked_tools = extract_invoked_tool_names(state)

            step_results[step.step_id] = {
                "agent": step.agent_name,
                "status": "completed" if step_score.total >= min_step_score else "failed",
                "score": step_score.total,
                "subtask": step.subtask,
                "raw_response": raw_response,
                "invoked_tools": invoked_tools,
            }
            if raw_response:
                response_parts.append(raw_response)

        return {
            "case_id": case.case_id,
            "user_query": case.user_query,
            "final_response": "\n".join(response_parts),
            "step_results": step_results,
        }

    @staticmethod
    def format_pipeline_output(result: Dict[str, Any]) -> str:
        """序列化 pipeline 输出供 MultiFieldEvaluation / 日志使用。"""
        payload = {
            "case_id": result.get("case_id"),
            "final_response": result.get("final_response"),
            "step_results": result.get("step_results"),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
