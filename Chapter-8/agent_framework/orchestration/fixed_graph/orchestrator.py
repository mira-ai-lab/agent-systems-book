"""LangGraph 版中心智能体编排器。

对外入口类 LangGraphOrchestrator：负责初始化 LLM、长期记忆、领域配置与编译后的 StateGraph，
并通过 process_request() 驱动完整工作流。

典型调用链：
    process_request(user_query)
        → app.ainvoke(initial_state)
        → 图节点依次更新 state
        → 返回 execution_plan / subtask_results / final_response
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from langchain_openai import ChatOpenAI

from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.registry import create_travel_registry, travel_domain_config
from agent_framework.domain.agent_factory import SubAgentFactory
from agent_framework.config import CHROMA_DIR, create_llm, load_project_dotenv
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import ContextBuilder, DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.infra.memory.memory_factory import create_long_term_memory, resolve_memory_backend
from agent_framework.tracing import get_logger, get_current_span_context, log_info, setup_observability, trace_span
from agent_framework.tracing.trace_provider import span_name

from .graph import compile_graph
from .state import CentralAgentState
from .stream_sink import StreamSink
from .visualize import GraphVisualizer

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """使用 LangGraph StateGraph 实现的中心智能体编排器。

    可注入 registry / prompts / domain_config / pipeline 以适配不同领域；
    未注入时默认使用旅行 demo（domains.travel）。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
        registry: Optional[SubAgentRegistry] = None,
        prompts: Optional[DomainPrompts] = None,
        domain_config: Optional[DomainConfig] = None,
        pipeline: Optional[PipelineConfig] = None,
        context_builder: Optional[ContextBuilder] = None,
        routing_fallback: Optional[str] = None,
        enable_guess_agent: bool = False,
    ) -> None:
        """初始化编排器并编译 LangGraph。

        参数说明：
            registry / prompts：领域子 Agent 与 prompt 包，默认旅行 demo。
            domain_config：context_builder（注入子任务 query）、guess 与路由策略。
            pipeline：是否启用 pre_survey / memory 节点。
            context_builder：快捷覆盖 domain_config.context_builder。
            routing_fallback / enable_guess_agent：路由失败时的兜底策略。
            enable_memory：与 pipeline.enable_memory 对齐，控制记忆读写。
        """
        load_project_dotenv()
        setup_observability()
        self.llm = llm or create_llm()
        self.registry = registry or create_travel_registry()
        SubAgentFactory.use_registry(self.registry)
        self.prompts = prompts or TravelPrompts.build()

        # 合并 domain_config：显式传入优先，其次快捷参数 routing_fallback / context_builder
        base_config = domain_config or travel_domain_config(enable_guess_agent=enable_guess_agent)
        if context_builder is not None:
            base_config = DomainConfig(
                context_builder=context_builder,
                guess_fn=base_config.guess_fn,
                routing_fallback=routing_fallback or base_config.routing_fallback,
                enable_guess_agent=enable_guess_agent or base_config.enable_guess_agent,
            )
        elif routing_fallback is not None:
            base_config = DomainConfig(
                context_builder=base_config.context_builder,
                guess_fn=base_config.guess_fn,
                routing_fallback=routing_fallback,
                enable_guess_agent=enable_guess_agent,
            )
        elif enable_guess_agent:
            base_config = DomainConfig(
                context_builder=base_config.context_builder,
                guess_fn=base_config.guess_fn,
                routing_fallback=base_config.routing_fallback,
                enable_guess_agent=True,
            )
        self.domain_config = base_config

        self.pipeline = pipeline or PipelineConfig(enable_memory=enable_memory)
        self.system_prompt = self.prompts.central_agent_system
        self.enable_memory = self.pipeline.enable_memory
        self.long_term_backend = resolve_memory_backend(long_term_backend)
        self.memory_system: Optional[Any] = None
        self.langgraph_store: Optional[Any] = None
        self.stream_sink = StreamSink()

        if self.enable_memory:
            try:
                self.memory_system, self.langgraph_store = create_long_term_memory(
                    self.long_term_backend,
                    user_id="central_agent_user",
                    llm=self.llm,
                    persist_directory=str(CHROMA_DIR),
                )
                backend_label = (
                    "LangGraph Store" if self.long_term_backend == "store" else "Chroma"
                )
                log_info(logger, "memory.enabled", backend=backend_label)
            except Exception as exc:
                # 记忆初始化失败时不阻断主流程，仅记录日志并继续无记忆模式
                log_info(logger, "memory.init_failed", error=str(exc))

        self.app = compile_graph(
            self.llm,
            self.memory_system,
            store=self.langgraph_store,
            stream_sink=self.stream_sink,
            registry=self.registry,
            prompts=self.prompts,
            domain_config=self.domain_config,
            pipeline=self.pipeline,
        )

    def _build_initial_state(
        self,
        user_query: str,
        thread_id: str,
        enable_stream: bool,
    ) -> CentralAgentState:
        """构造图初始 state，含 query、thread_id、记忆/流式开关等。"""
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
        """将 token / 进度回调绑定到 stdout，供 CLI 流式演示使用。"""
        self.stream_sink.enabled = True

        def _write_token(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        self.stream_sink.on_token = _write_token
        self.stream_sink.on_progress = lambda msg: print(msg, flush=True)

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """批量模式：一次性执行完整图，返回规划、子任务结果与最终回复。"""
        log_info(logger, "request.start", thread_id=thread_id, query_preview=user_query.strip()[:120])
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

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request_stream(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """流式模式：阶段进度 + 聚合 LLM 的 token 实时输出到 stdout。"""
        self._attach_stdout_stream_handlers()
        try:
            print("\n" + "=" * 60)
            print(f"📝 用户：{user_query.strip()}")
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
            # stream_mode="updates"：逐节点推送局部 state 更新
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
        name=span_name("request.stream"),
        attrs_args=["user_query", "thread_id"],
        record_result=False,
    )
    async def iter_request_stream(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """流式迭代器：供 Web / 自定义 UI 消费图更新事件，不绑定 stdout。"""
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
        """从最终 state 提取对外 API 统一返回结构。"""
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
        """返回图可视化工具实例。"""
        return GraphVisualizer.from_compiled(self.app)

    def get_graph_mermaid(self) -> str:
        """导出 Mermaid 源码，可粘贴到 mermaid.live 预览。"""
        return self.get_visualizer().get_mermaid()

    def show_graph(self) -> None:
        """在终端打印图结构（文本 + Mermaid + ASCII）。"""
        self.get_visualizer().print_all()

    def save_graph(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        """保存 .mmd / .png / .txt 到 output 目录。"""
        return self.get_visualizer().save_all(output_dir, prefix)
