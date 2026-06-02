"""Supervisor 版中心智能体 — 与 langgraph/ 功能等价（动态调度模式）"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

SUP_DIR = Path(__file__).resolve().parent
CHAPTER6_DIR = SUP_DIR.parent
if str(SUP_DIR) not in sys.path:
    sys.path.insert(0, str(SUP_DIR))
if str(CHAPTER6_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER6_DIR))

load_dotenv(CHAPTER6_DIR.parent / ".env")

from _ch6_loader import load_ch6_module
from agents import AGENT_SPECS
from memory_factory import LongTermBackend, create_long_term_memory, resolve_memory_backend
from supervisor_graph import build_supervisor_app

_aggregation = load_ch6_module("aggregation_helpers")
_central = load_ch6_module("central_orchestrator")
_planner = load_ch6_module("task_planner")

TaskPlanner = _planner.TaskPlanner
SubAgentRegistry = _central.SubAgentRegistry
is_single_direct_response = _aggregation.is_single_direct_response
direct_response_from_results = _aggregation.direct_response_from_results


def _last_ai_content(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return str(msg.content).strip()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content).strip()
    return ""


class SupervisorOrchestrator:
    """
    LangGraph Supervisor 模式中心智能体

    记忆：
    - 短期：MemorySaver checkpoint（Supervisor 原生 thread 记忆）
    - 长期：可切换
        - long_term_backend="chroma" → Chroma（Chapter-3，持久化到 chroma_memory/）
        - long_term_backend="store"   → LangGraph Store（语义检索，注入 compile(store=...)）
    - 环境变量 MEMORY_BACKEND=chroma|store 可作为默认值
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[LongTermBackend, str]] = None,
    ) -> None:
        self.llm = llm or self._create_default_llm()
        self.enable_memory = enable_memory
        self.long_term_backend = resolve_memory_backend(long_term_backend)
        self.memory_system: Optional[Any] = None
        self.langgraph_store: Optional[Any] = None

        if enable_memory:
            try:
                self.memory_system, self.langgraph_store = create_long_term_memory(
                    self.long_term_backend,
                    user_id="supervisor_user",
                    llm=self.llm,
                    persist_directory=str(CHAPTER6_DIR / "chroma_memory"),
                )
                backend_label = (
                    "LangGraph Store"
                    if self.long_term_backend == "store"
                    else "Chroma"
                )
                print(f"✓ 长期记忆已启用（{backend_label}）", flush=True)
            except Exception as exc:
                print(f"⚠️ 长期记忆初始化失败: {exc}", flush=True)

        self.planner = TaskPlanner(self.llm, SubAgentRegistry())
        self.app = build_supervisor_app(self.llm, store=self.langgraph_store)

    def _create_default_llm(self) -> ChatOpenAI:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        return ChatOpenAI(
            model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
            temperature=0,
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_CHAT_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            http_client=httpx.Client(verify=False),
        )

    def _build_initial_message(self, user_query: str, thread_id: str) -> str:
        parts = [user_query.strip()]

        if self.memory_system and self.enable_memory:
            hits = self.memory_system.search_memories(user_query)
            memories = self.memory_system.format_memories_for_plan(hits)
            if memories:
                parts.append("\n【相关长期记忆】")
                for m in memories:
                    line = m.get("summary") or m.get("content") or str(m)
                    kps = m.get("key_points") or []
                    if kps:
                        line += f" | 要点: {', '.join(map(str, kps))}"
                    parts.append(f"- {line}")
            stm = self.memory_system.short_term.format_recent(thread_id, last_n=4)
            if stm and stm != "（无历史对话）":
                parts.append(f"\n【本会话近期对话】\n{stm}")

        return "\n".join(parts)

    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        print("=" * 80, flush=True)
        print(f"📥 [Supervisor] 用户请求: {user_query.strip()}", flush=True)
        print("=" * 80, flush=True)

        print("\n🔍 [Ch2] 思维链预调查...", flush=True)
        pre_survey = await self.planner.run_pre_survey(user_query)
        print("✓ 预调查完成", flush=True)

        if self.memory_system and self.enable_memory:
            hits = self.memory_system.search_memories(user_query)
            mem_count = len(self.memory_system.format_memories_for_plan(hits))
            backend = getattr(self.memory_system, "backend_name", self.long_term_backend)
            print(f"\n🧠 [Ch3] 长期记忆({backend}) 检索到 {mem_count} 条", flush=True)
        else:
            print("\n🧠 [Ch3] 长期记忆已跳过", flush=True)

        enriched_query = self._build_initial_message(user_query, thread_id)
        if pre_survey:
            summary = {k: v for k, v in pre_survey.items() if k != "raw_text"}
            enriched_query += f"\n\n【预调查摘要】\n{summary}"

        print("\n🎯 [Supervisor] 动态调度子智能体...", flush=True)
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=enriched_query)]},
            config,
        )

        messages = result.get("messages", [])
        final_response = _last_ai_content(messages)

        subtask_results = self._extract_subtask_results(messages)
        if is_single_direct_response(subtask_results):
            direct = direct_response_from_results(subtask_results)
            if direct:
                final_response = direct
            print("  ✓ 单任务查询，直接使用子智能体回复", flush=True)

        if self.memory_system and self.enable_memory:
            self.memory_system.record_turn(thread_id, user_query, final_response)
            await self.memory_system.ingest(
                f"用户请求: {user_query.strip()}\n偏好摘要: {final_response[:500]}",
                memory_type="preference",
            )
            backend = getattr(self.memory_system, "backend_name", self.long_term_backend)
            print(f"\n💾 [Ch3] 已写入长期记忆({backend})", flush=True)
        else:
            print("\n💾 [Ch3] 记忆写入已跳过", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("✅ [Supervisor] 全部处理完成", flush=True)
        print("=" * 80, flush=True)

        return {
            "pre_survey": pre_survey,
            "subtask_results": subtask_results,
            "final_response": final_response,
            "messages": messages,
            "long_term_backend": self.long_term_backend,
        }

    def _extract_subtask_results(self, messages: List[Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        idx = 0
        agent_names = {n for n, _, _ in AGENT_SPECS}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.name in agent_names:
                idx += 1
                tid = f"T{idx}"
                results[tid] = {
                    "task_id": tid,
                    "agent": msg.name,
                    "status": "completed",
                    "agent_summary": str(msg.content or ""),
                }
        return results

    def get_graph_mermaid(self) -> str:
        try:
            return self.app.get_graph().draw_mermaid()
        except Exception:
            return "graph TD\n  supervisor --> sub_agents"
