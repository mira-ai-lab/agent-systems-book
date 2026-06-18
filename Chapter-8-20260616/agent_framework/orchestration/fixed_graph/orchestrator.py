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

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from langchain_openai import ChatOpenAI

from agent_framework.domain.agent_factory import SubAgentFactory
from agent_framework.config import CHROMA_DIR, REQUEST_TIMEOUT_SEC, REQUEST_SLOT_WAIT_SEC, create_llm, load_project_dotenv
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import ContextBuilder, DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.infra.memory.memory_factory import create_long_term_memory, resolve_memory_backend
from agent_framework.infra.checkpoint_factory import resolve_checkpointer
from agent_framework.infra.concurrency import acquire_request_slot
from agent_framework.i18n.agent_locale_context import agent_locale_context
from agent_framework.observability.request_context import request_metrics_context
from agent_framework.orchestration.protocol import MODE_FIXED_GRAPH
from agent_framework.stream.events import (
    final_event,
    graph_node_event,
    graph_progress_event,
    graph_subtask_completed_event,
    graph_subtask_token_event,
    graph_token_event,
)
from agent_framework.tracing import get_logger, get_current_span_context, log_info, setup_observability, trace_span
from agent_framework.tracing.trace_provider import span_name

from .graph import compile_graph
from .state import CentralAgentState
from .stream_sink import StreamSink
from .visualize import GraphVisualizer

logger = get_logger(__name__)


def _resolve_domain_bundle(
    domain: str,
    *,
    enable_guess_agent: bool = False,
) -> tuple[SubAgentRegistry, DomainPrompts, DomainConfig]:
    from agent_framework.domain.plugin_registry import get_domain_plugin

    plugin = get_domain_plugin(domain)
    return (
        plugin.create_registry(),
        plugin.create_prompts(),
        plugin.create_domain_config(enable_guess_agent=enable_guess_agent),
    )


class LangGraphOrchestrator:
    """使用 LangGraph StateGraph 实现的中心智能体编排器。

    必须通过 ``domain=...`` 或同时注入 ``registry`` / ``prompts`` / ``domain_config``；
    框架层不再默认回落到旅行领域。推荐入口：``create_orchestrator(domain)``。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
        domain: Optional[str] = None,
        registry: Optional[SubAgentRegistry] = None,
        prompts: Optional[DomainPrompts] = None,
        domain_config: Optional[DomainConfig] = None,
        pipeline: Optional[PipelineConfig] = None,
        context_builder: Optional[ContextBuilder] = None,
        routing_fallback: Optional[str] = None,
        enable_guess_agent: bool = False,
        user_id: str = "default",
        locale: str = "zh",
    ) -> None:
        """初始化编排器并编译 LangGraph。

        参数说明：
            domain：已注册领域名（与显式注入三件套二选一）。
            registry / prompts / domain_config：领域子 Agent、prompt 与路由策略。
            pipeline：是否启用 pre_survey / memory 节点。
            context_builder：快捷覆盖 domain_config.context_builder。
            routing_fallback / enable_guess_agent：路由失败时的兜底策略。
            enable_memory：与 pipeline.enable_memory 对齐，控制记忆读写。
            user_id：多租户长期记忆隔离键。
        """
        load_project_dotenv()
        setup_observability()
        self.user_id = (user_id or "default").strip() or "default"
        self.locale = (locale or "zh").strip() or "zh"
        self.request_timeout_sec = REQUEST_TIMEOUT_SEC
        self.domain = (domain or "").strip() or None
        self.mode = MODE_FIXED_GRAPH

        if registry is None or prompts is None or domain_config is None:
            partial = sum(x is not None for x in (registry, prompts, domain_config))
            if partial not in (0, 3):
                raise ValueError(
                    "registry / prompts / domain_config 必须同时注入，"
                    "或使用 domain=... 由插件解析"
                )
            if not self.domain:
                raise ValueError(
                    "缺少领域配置：请传入 domain='customer_service' 等已注册领域，"
                    "或 travel（书稿示例）。"
                    "推荐：agent_framework.bootstrap.platform.create_orchestrator(domain)"
                )
            resolved_registry, resolved_prompts, resolved_domain_config = _resolve_domain_bundle(
                self.domain,
                enable_guess_agent=enable_guess_agent,
            )
            registry = registry or resolved_registry
            prompts = prompts or resolved_prompts
            domain_config = domain_config or resolved_domain_config

        self.llm = llm or create_llm()

        from agent_framework.infra.agent_runtime import configure_agent_llm

        configure_agent_llm(self.llm)

        self.registry = registry
        SubAgentFactory.use_registry(self.registry)
        self.prompts = prompts

        # 合并 domain_config：显式传入优先，其次快捷参数 routing_fallback / context_builder
        base_config = domain_config
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
                    user_id=self.user_id,
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
            checkpointer=resolve_checkpointer(),
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
        *,
        prefilled_execution_plan: Optional[Dict[str, Any]] = None,
    ) -> CentralAgentState:
        """构造图初始 state，含 query、thread_id、记忆/流式开关等。"""
        state: CentralAgentState = {
            "user_query": user_query,
            "thread_id": thread_id,
            "enable_memory": self.enable_memory,
            "enable_stream": enable_stream,
            "logs": [],
            "subtask_results": {},
            "current_layer_index": 0,
        }
        if prefilled_execution_plan:
            state["prefilled_execution_plan"] = prefilled_execution_plan
            router_pre = prefilled_execution_plan.get("pre_survey") or {}
            if router_pre.get("source") == "router_engine":
                state["prefilled_pre_survey"] = router_pre
        return state

    def _attach_stdout_stream_handlers(self) -> None:
        """将 token / 进度回调绑定到 stdout，供 CLI 流式演示使用。"""
        from agent_framework.stream.events import build_subtask_summary

        self.stream_sink.enabled = True

        def _write_token(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        subtask_headers: set[tuple[str, str]] = set()

        def _print_subtask_token(task_id: str, agent: str, token: str) -> None:
            key = (task_id, agent)
            if key not in subtask_headers:
                print(f"\n[{task_id} → {agent}] ", end="", flush=True)
                subtask_headers.add(key)
            print(token, end="", flush=True)

        def _print_subtask_done(result: Dict[str, Any]) -> None:
            task_id = str(result.get("task_id") or "?")
            agent = str(result.get("agent") or "?")
            key = (task_id, agent)
            if key in subtask_headers:
                print(f"\n✓ [{task_id} → {agent}] done", flush=True)
            else:
                summary = build_subtask_summary(
                    result,
                    summarizer=self.domain_config.subtask_summary_fn,
                )
                print(f"\n✓ [{task_id} → {agent}] {summary}", flush=True)

        self.stream_sink.on_token = _write_token
        self.stream_sink.on_progress = lambda msg: print(msg, flush=True)
        self.stream_sink.on_subtask_completed = _print_subtask_done
        self.stream_sink.on_subtask_token = _print_subtask_token

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
        timeout_sec: Optional[float] = None,
        *,
        prefilled_execution_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """批量模式：一次性执行完整图，返回规划、子任务结果与最终回复。"""
        log_info(logger, "request.start", thread_id=thread_id, query_preview=user_query.strip()[:120])
        initial_state = self._build_initial_state(
            user_query,
            thread_id,
            enable_stream=False,
            prefilled_execution_plan=prefilled_execution_plan,
        )
        config = {"configurable": {"thread_id": thread_id}}
        deadline = timeout_sec if timeout_sec is not None else self.request_timeout_sec
        with agent_locale_context(self.locale):
            with request_metrics_context(
                domain=self.domain or "",
                mode=MODE_FIXED_GRAPH,
                transport="local",
            ):
                async with acquire_request_slot(wait_timeout_sec=REQUEST_SLOT_WAIT_SEC):
                    final_state = await asyncio.wait_for(
                        self.app.ainvoke(initial_state, config),
                        timeout=deadline,
                    )
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
        *,
        prefilled_execution_plan: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """流式迭代器：节点执行期间通过 Queue 实时 yield，不等到 LangGraph 节点结束。"""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _DONE = object()

        def _enqueue(event: Dict[str, Any]) -> None:
            queue.put_nowait(event)

        def on_progress(message: str) -> None:
            _enqueue(graph_progress_event(message))

        def on_token(text: str) -> None:
            if text:
                _enqueue(graph_token_event(text))

        def on_subtask_completed(result: Dict[str, Any]) -> None:
            _enqueue(
                graph_subtask_completed_event(
                    result,
                    summarizer=self.domain_config.subtask_summary_fn,
                )
            )

        def on_subtask_token(task_id: str, agent: str, token: str) -> None:
            _enqueue(graph_subtask_token_event(task_id, agent, token))

        self.stream_sink.enabled = True
        self.stream_sink.on_progress = on_progress
        self.stream_sink.on_token = on_token
        self.stream_sink.on_subtask_completed = on_subtask_completed
        self.stream_sink.on_subtask_token = on_subtask_token

        async def _run_graph(initial_state: CentralAgentState, config: Dict[str, Any]) -> None:
            try:
                async for chunk in self.app.astream(
                    initial_state,
                    config,
                    stream_mode="updates",
                ):
                    _enqueue(graph_node_event(chunk))
            finally:
                queue.put_nowait(_DONE)

        try:
            with agent_locale_context(self.locale):
                initial_state = self._build_initial_state(
                    user_query,
                    thread_id,
                    enable_stream=True,
                    prefilled_execution_plan=prefilled_execution_plan,
                )
                config = {"configurable": {"thread_id": thread_id}}
                graph_task = asyncio.create_task(_run_graph(initial_state, config))
                while True:
                    item = await queue.get()
                    if item is _DONE:
                        break
                    yield item
                await graph_task
                snapshot = await self.app.aget_state(config)
                final_state = dict(snapshot.values) if snapshot and snapshot.values else {}
                trace_id, span_id = get_current_span_context()
                yield final_event(self._result_from_state(final_state, trace_id, span_id))
        finally:
            self.stream_sink.reset()

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
