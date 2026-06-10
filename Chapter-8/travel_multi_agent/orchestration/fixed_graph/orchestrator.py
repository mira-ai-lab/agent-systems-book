"""LangGraph 版中心智能体编排器"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain_openai import ChatOpenAI

from travel_multi_agent.config import CHROMA_DIR, create_llm, load_project_dotenv
from travel_multi_agent.domain.prompts import CENTRAL_AGENT_SYSTEM_PROMPT
from travel_multi_agent.infra.memory.memory_factory import create_long_term_memory, resolve_memory_backend
from travel_multi_agent.tracing import get_logger, get_trace_ids, log_info, setup_observability, span

from .graph import compile_graph
from .state import CentralAgentState
from .visualize import GraphVisualizer

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """使用 LangGraph StateGraph 实现的中心智能体。"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_memory: bool = True,
        long_term_backend: Optional[Union[str, Any]] = None,
    ) -> None:
        load_project_dotenv()
        setup_observability()
        self.llm = llm or create_llm()
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
        )

    async def process_request(
        self,
        user_query: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """运行 LangGraph 工作流"""
        with span(
            "travel.request",
            **{
                "thread.id": thread_id,
                "user.query_length": len(user_query.strip()),
            },
        ):
            log_info(
                logger,
                "request.start",
                thread_id=thread_id,
                query_preview=user_query.strip()[:120],
            )

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

            trace_id, span_id = get_trace_ids()
            log_info(
                logger,
                "request.done",
                thread_id=thread_id,
                trace_id=trace_id,
                span_id=span_id,
                subtask_count=len(final_state.get("subtask_results") or {}),
            )

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
        return GraphVisualizer.from_compiled(self.app)

    def get_graph_mermaid(self) -> str:
        return self.get_visualizer().get_mermaid()

    def show_graph(self) -> None:
        self.get_visualizer().print_all()

    def save_graph(
        self,
        output_dir: Optional[Path] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        return self.get_visualizer().save_all(output_dir, prefix)
