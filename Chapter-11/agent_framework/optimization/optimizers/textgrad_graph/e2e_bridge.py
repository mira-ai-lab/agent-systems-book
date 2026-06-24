"""E2E 编排器同步桥接：在 textgrad StringBasedFunction 中复用 Router/LangGraph 路径。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.e2e.runtime import build_e2e_orchestrator


class E2eOrchestratorSyncBridge:
    """将 ``process_request`` 包装为 sync 调用，prompt 由 Variable 每次注入。"""

    def __init__(
        self,
        *,
        executor_llm: ChatOpenAI,
        locale: str = "zh",
        profile: str = "workflow",
        enable_memory: bool = False,
        enable_guess_agent: bool = True,
        timeout_sec: Optional[float] = None,
    ):
        self._executor_llm = executor_llm
        self._locale = locale
        self._profile = profile
        self._enable_memory = enable_memory
        self._enable_guess_agent = enable_guess_agent
        self._timeout_sec = timeout_sec

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    def process_request(
        self,
        *,
        decomposition_prompt: str,
        agent_routing: str,
        user_query: str,
        thread_id: str,
    ) -> Dict[str, Any]:
        orchestrator = build_e2e_orchestrator(
            self._executor_llm,
            locale=self._locale,
            profile=self._profile,
            enable_memory=self._enable_memory,
            enable_guess_agent=self._enable_guess_agent,
            prompt_overrides={
                "decomposition_prompt": decomposition_prompt,
                "agent_routing": agent_routing,
            },
            use_optimized=False,
        )
        return self._run_async(
            orchestrator.process_request(
                user_query,
                thread_id=thread_id,
                timeout_sec=self._timeout_sec,
            )
        )

    @staticmethod
    def format_e2e_output(result: Dict[str, Any]) -> str:
        """序列化 E2E 结果供 MultiFieldEvaluation 与日志使用。"""
        subtask_results = result.get("subtask_results") or {}
        invoked = sorted(
            {
                str(item.get("agent") or "").strip()
                for item in subtask_results.values()
                if isinstance(item, dict) and str(item.get("agent") or "").strip()
            }
        )
        completed = sum(
            1
            for item in subtask_results.values()
            if isinstance(item, dict)
            and str(item.get("status") or "").lower() in ("completed", "ok")
        )
        payload = {
            "final_response": str(result.get("final_response") or ""),
            "invoked_agents": invoked,
            "completed_subtasks": completed,
            "subtask_results": subtask_results,
            "orchestration_mode": str(result.get("orchestration_mode") or ""),
            "trace_id": str(result.get("trace_id") or ""),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
