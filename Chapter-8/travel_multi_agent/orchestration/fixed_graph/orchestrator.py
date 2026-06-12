"""LangGraph 版中心智能体编排器。

对外入口类 LangGraphOrchestrator：负责初始化 LLM、长期记忆与编译后的 StateGraph，
并通过 process_request() 驱动预调查 → 规划 → 分层执行 → 聚合的完整工作流。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from langchain_openai import ChatOpenAI

from travel_multi_agent.config import CHROMA_DIR, create_llm, load_project_dotenv
from travel_multi_agent.domain.prompts import CENTRAL_AGENT_SYSTEM_PROMPT
from travel_multi_agent.infra.memory.memory_factory import create_long_term_memory, resolve_memory_backend
from travel_multi_agent.tracing import get_logger, get_current_span_context, log_info, setup_observability, trace_span

from .graph import compile_graph
from .state import CentralAgentState
from .stream_sink import StreamSink
from .visualize import GraphVisualizer

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """使用 LangGraph StateGraph 实现的中心智能体。

    属性 self.app 为 compile_graph() 产出的可调用图，供 process_request 异步执行。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
    ) -> None:
        """初始化编排器：加载环境变量、观测性、LLM、记忆后端，并编译 LangGraph。

        long_term_backend 可选 "store"（LangGraph Store）或 Chroma；记忆初始化失败时
        仅记录日志，不阻断图编译，对应节点会在运行时跳过记忆读写。
        """
        load_project_dotenv()
        setup_observability()
        self.llm = llm or create_llm()
        self.system_prompt = CENTRAL_AGENT_SYSTEM_PROMPT
        self.enable_memory = enable_memory
        self.long_term_backend = resolve_memory_backend(long_term_backend)
        self.memory_system: Optional[Any] = None
        self.langgraph_store: Optional[Any] = None
        self.stream_sink = StreamSink()

        if enable_memory:
            try:
                self.memory_system, self.langgraph_store = create_long_term_memory(
                    self.long_term_backend,
                    user_id="central_agent_user",
                    llm=self.llm,
                    persist_directory=str(CHROMA_DIR),
                )
                backend_label = (
                    "LangGraph Store"
                    if self.long_term_backend == "store"
                    else "Chroma"
                )
                log_info(logger, "memory.enabled", backend=backend_label)
            except Exception as exc:
                log_info(logger, "memory.init_failed", error=str(exc))

        self.app = compile_graph(
            self.llm,
            self.memory_system,
            store=self.langgraph_store,
            stream_sink=self.stream_sink,
        )

    def _build_initial_state(
        self,
        user_query: str,
        thread_id: str,
        enable_stream: bool,
    ) -> CentralAgentState:
        return {
            "user_query": user_query,
            "thread_id": thread_id,
            "enable_memory": self.enable_memory,
            "enable_stream": enable_stream,
            "logs": [],
            "subtask_results": {},
            "current_layer_index": 0,
        }

    def _attach_stdout_stream_handlers(self) -> None:
        """将 token / 进度输出绑定到标准输出。"""
        self.stream_sink.enabled = True

        def _write_token(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        self.stream_sink.on_token = _write_token
        self.stream_sink.on_progress = lambda msg: print(msg, flush=True)

    @trace_span(
        name="latc.travel-multi-agent.request",
        attrs_args=["user_query", "thread_id"],
    )
    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """运行完整 LangGraph 工作流，返回执行计划、子任务结果与最终回复。

        thread_id 用于 LangGraph checkpoint 与子 Agent 会话隔离；同一 thread_id 可复用上下文。
        返回字典含 execution_plan / subtask_results / final_response / logs /
        graph_state（完整终态）/ trace_id / span_id。
        """
        log_info(
            logger,
            "request.start",
            thread_id=thread_id,
            query_preview=user_query.strip()[:120],
        )

        initial_state = self._build_initial_state(user_query, thread_id, enable_stream=False)
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.app.ainvoke(initial_state, config)

        trace_id, span_id = get_current_span_context()
        log_info(
            logger,
            "request.done",
            thread_id=thread_id,
            trace_id=trace_id,
            span_id=span_id,
            subtask_count=len(final_state.get("subtask_results") or {}),
        )

        return self._result_from_state(final_state, trace_id, span_id)

    @trace_span(
        name="latc.travel-multi-agent.request",
        attrs_args=["user_query", "thread_id"],
    )
    async def process_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """流式运行工作流：阶段进度 + 聚合 LLM 逐 token 输出到 stdout。

        与 process_request 返回结构相同；适合 CLI 演示与交互对话。
        """
        self._attach_stdout_stream_handlers()
        try:
            print("\n" + "=" * 60)
            print(f"📥 用户：{user_query.strip()}")
            print("=" * 60)

            log_info(
                logger,
                "request.start",
                thread_id=thread_id,
                query_preview=user_query.strip()[:120],
                stream=True,
            )

            initial_state = self._build_initial_state(user_query, thread_id, enable_stream=True)
            config = {"configurable": {"thread_id": thread_id}}

            async for _ in self.app.astream(initial_state, config, stream_mode="updates"):
                pass

            snapshot = await self.app.aget_state(config)
            final_state = dict(snapshot.values) if snapshot and snapshot.values else {}

            trace_id, span_id = get_current_span_context()
            log_info(
                logger,
                "request.done",
                thread_id=thread_id,
                trace_id=trace_id,
                span_id=span_id,
                subtask_count=len(final_state.get("subtask_results") or {}),
                stream=True,
            )
            return self._result_from_state(final_state, trace_id, span_id)
        finally:
            self.stream_sink.reset()

    @trace_span(
        name="latc.travel-multi-agent.request.stream",
        attrs_args=["user_query", "thread_id"],
        record_result=False,
    )
    async def iter_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步迭代图节点更新（供 Web / 自定义 UI 消费）。"""
        initial_state = self._build_initial_state(user_query, thread_id, enable_stream=True)
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self.app.astream(initial_state, config, stream_mode="updates"):
            yield chunk

    @staticmethod
    def _result_from_state(
        final_state: Dict[str, Any],
        trace_id: Optional[str],
        span_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "execution_plan": final_state.get("execution_plan"),
            "subtask_results": final_state.get("subtask_results"),
            "final_response": final_state.get("final_response", ""),
            "logs": final_state.get("logs", []),
            "graph_state": final_state,
            "trace_id": trace_id,
            "span_id": span_id,
        }

    def get_visualizer(self) -> GraphVisualizer:
        """基于已编译图创建可视化器，用于导出或打印节点/边结构。"""
        return GraphVisualizer.from_compiled(self.app)

    def get_graph_mermaid(self) -> str:
        """返回 Mermaid 格式的图结构字符串，便于文档或前端渲染。"""
        return self.get_visualizer().get_mermaid()

    def show_graph(self) -> None:
        """在终端打印图结构（文本 + Mermaid）。"""
        self.get_visualizer().print_all()

    def save_graph(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        """将图导出为 .mmd / .png / .txt 等文件，返回各格式对应的文件路径。"""
        return self.get_visualizer().save_all(output_dir, prefix)
