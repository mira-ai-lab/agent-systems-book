"""LangGraph central orchestrator entry point.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from langchain_openai import ChatOpenAI

from travel_multi_agent.agents.registry import DomainAgentRegistry, create_travel_registry
from travel_multi_agent.a2a.client import A2ARegistry
from travel_multi_agent.config import AGENT_CARDS_DIR, CHROMA_DIR, load_project_dotenv
from llm import create_llm
from travel_multi_agent.orchestration.pipeline import PipelineConfig
from travel_multi_agent.memory import create_long_term_memory, resolve_memory_backend
from travel_multi_agent.tracing import get_logger, get_current_span_context, log_info, setup_observability, trace_span
from travel_multi_agent.tracing.trace_provider import span_name

from .graph import compile_graph
from .state import CentralAgentState
from .stream_sink import StreamSink
from .visualize import GraphVisualizer

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """Central agent implemented with a LangGraph StateGraph.

    ``registry`` / ``a2a_registry`` / ``pipeline`` may be injected to adapt to
    another domain; without them the travel demo (``travel_multi_agent``) is
    used. Domain agents are separate A2A services registered via Agent Cards.
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
        registry: Optional[DomainAgentRegistry] = None,
        a2a_registry: Optional[A2ARegistry] = None,
        pipeline: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize and compile the LangGraph orchestrator."""
        load_project_dotenv()
        setup_observability()
        self.llm = llm or create_llm()
        self.registry = registry or create_travel_registry()
        self.a2a_registry = a2a_registry or A2ARegistry.from_directory(AGENT_CARDS_DIR)
        self.pipeline = pipeline or PipelineConfig(enable_memory=enable_memory)
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
                # memory init failure must not block the main flow
                log_info(logger, "memory.init_failed", error=str(exc))

        self.app = compile_graph(
            self.llm,
            self.memory_system,
            store=self.langgraph_store,
            stream_sink=self.stream_sink,
            registry=self.registry,
            a2a_registry=self.a2a_registry,
            pipeline=self.pipeline,
        )

    def _build_initial_state(
        self,
        user_query: str,
        thread_id: str,
        enable_stream: bool,
    ) -> CentralAgentState:
        """Build the graph initial state for one request."""
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
        """Bind token / progress callbacks to stdout for the CLI demo."""
        self.stream_sink.enabled = True

        def _write_token(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        self.stream_sink.on_token = _write_token
        self.stream_sink.on_progress = lambda msg: print(msg, flush=True)

    @trace_span(name=span_name("request"), attrs_args=["user_query", "thread_id"])
    async def process_request(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Batch mode: run the full graph once and return the structured result."""
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
        """Streaming mode: stage progress plus the final response to stdout."""
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
        """Async iterator of graph update events for web / custom UIs."""
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
        """Extract the stable public result structure from the final state."""
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
        """Return the graph visualizer for this compiled app."""
        return GraphVisualizer.from_compiled(self.app)

    def get_graph_mermaid(self) -> str:
        """Export Mermaid source for mermaid.live."""
        return self.get_visualizer().get_mermaid()

    def show_graph(self) -> None:
        """Print the graph structure (text + Mermaid + ASCII)."""
        self.get_visualizer().print_all()

    def save_graph(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        """Save .mmd / .png / .txt to the output directory."""
        return self.get_visualizer().save_all(output_dir, prefix)