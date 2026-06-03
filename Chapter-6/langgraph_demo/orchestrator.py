"""LangGraph 版中心智能体编排器"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import httpx

LG_DIR = Path(__file__).resolve().parent
CHAPTER6_DIR = LG_DIR.parent
if str(LG_DIR) not in sys.path:
    sys.path.insert(0, str(LG_DIR))

load_dotenv(CHAPTER6_DIR.parent / ".env")

from _ch6_loader import load_ch6_module
from graph import compile_graph
from state import CentralAgentState
from visualize import GraphVisualizer

_memory_factory = load_ch6_module("memory_factory")
_prompts = load_ch6_module("prompts")
create_long_term_memory = _memory_factory.create_long_term_memory
resolve_memory_backend = _memory_factory.resolve_memory_backend
CENTRAL_AGENT_SYSTEM_PROMPT = _prompts.CENTRAL_AGENT_SYSTEM_PROMPT


class LangGraphOrchestrator:
    """
    使用 LangGraph StateGraph 实现的中心智能体

    与 central_orchestrator.CentralOrchestrator 功能等价，
    工作流显式建模为图节点 + 条件边，支持 MemorySaver checkpoint。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
    ) -> None:
        self.llm = llm or self._create_default_llm()
        self.system_prompt = CENTRAL_AGENT_SYSTEM_PROMPT
        self.enable_memory = enable_memory
        self.long_term_backend = resolve_memory_backend(long_term_backend)
        self.memory_system: Optional[Any] = None
        self.langgraph_store: Optional[Any] = None

        if enable_memory:
            try:
                self.memory_system, self.langgraph_store = create_long_term_memory(
                    self.long_term_backend,
                    user_id="central_agent_user",
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

        self.app = compile_graph(
            self.llm,
            self.memory_system,
            store=self.langgraph_store,
        )

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

    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """运行 LangGraph 工作流"""
        print("=" * 80, flush=True)
        print(f"📥 [LangGraph] 用户请求: {user_query.strip()}", flush=True)
        print("=" * 80, flush=True)

        initial_state: CentralAgentState = {
            "user_query": user_query,
            "thread_id": thread_id,
            "enable_memory": self.enable_memory,
            "logs": [],
            "subtask_results": {},
            "current_layer_index": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.app.ainvoke(initial_state, config)

        print("\n" + "=" * 80, flush=True)
        print("✅ [LangGraph] 全部处理完成", flush=True)
        print("=" * 80, flush=True)

        return {
            "execution_plan": final_state.get("execution_plan"),
            "subtask_results": final_state.get("subtask_results"),
            "final_response": final_state.get("final_response", ""),
            "logs": final_state.get("logs", []),
            "graph_state": final_state,
        }

    def get_visualizer(self) -> GraphVisualizer:
        """返回图可视化工具（Mermaid / ASCII / PNG）"""
        return GraphVisualizer.from_compiled(self.app)

    def get_graph_mermaid(self) -> str:
        """返回 Mermaid 图源码"""
        return self.get_visualizer().get_mermaid()

    def show_graph(self) -> None:
        """在终端打印完整图结构（节点说明 + ASCII + Mermaid）"""
        self.get_visualizer().print_all()

    def save_graph(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        """保存图结构到 output/（.mmd / .txt / .png）"""
        return self.get_visualizer().save_all(output_dir, prefix)
